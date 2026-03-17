"""AI image generation (Gemini/OpenAI) mixin."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any

from ._constants import (
    AI_FIGURE_TEMPLATES,
    _clean_ai_image_caption,
    FIGURE_PROMPT_SYSTEM,
    MAX_IMAGE_PROMPT_LEN,
    MAX_IMAGE_RETRIES,
    MAX_OPTIMIZED_PROMPT_LEN,
    PROMPT_CORE_PRINCIPLES,
)

logger = logging.getLogger(__name__)


class _AiFigureMixin:
    """Mixin — AI-generated figure methods."""

    async def _generate_ai_figure(
        self,
        context: str,
        fig_key: str,
        filename_stem: str,
        description: str,
        ai_image_type: str = "generic",
        caption: str = "",
    ) -> dict[str, Any]:
        """Generate a single figure via AI image model (Gemini).

        Flow: generate prompt → try Gemini (with retries) → if all fail,
        LLM diagnoses error & optimizes prompt → retry → fallback to code chart.
        """
        # Ensure the caption used for the paper is short & academic, not the
        # verbose image-generation prompt.  The *description* is meant for
        # prompt generation; the *caption* goes into LaTeX.
        clean_caption = _clean_ai_image_caption(
            caption or description, title=fig_key.replace("_", " ").title(),
        )

        # Look up the template for this AI figure type
        template = AI_FIGURE_TEMPLATES.get(ai_image_type, AI_FIGURE_TEMPLATES["generic"])

        # Step 1: LLM generates image prompt using the template as a reference
        user_prompt = (
            f"Research context:\n{context}\n\n"
            f"Figure description:\n{description}\n\n"
            f"=== REFERENCE TEMPLATE (adapt to match the research context above) ===\n"
            f"{template}\n"
            f"=== END TEMPLATE ===\n\n"
            f"{PROMPT_CORE_PRINCIPLES}\n\n"
            f"Write a DETAILED image generation prompt for this specific figure.\n"
            f"Use the reference template as a STRUCTURAL GUIDE, but customize ALL content\n"
            f"to match the actual research: replace generic module names with the real\n"
            f"component names, use the actual method name, datasets, and metrics.\n\n"
            f"REQUIREMENTS:\n"
            f"- The figure must look like it belongs in a NeurIPS/ICML/CVPR paper\n"
            f"- Describe the EXACT spatial layout: what goes where, data flow direction\n"
            f"- Specify colors by hex code (use academic-standard muted tones, max 4 hues)\n"
            f"- Name every component/module/block with its actual research name\n"
            f"- Describe arrow routing and data flow directions explicitly\n"
            f"- Include tensor dimension annotations where relevant (e.g., B×L×D)\n"
            f"- Mark the NOVEL components with a distinct visual treatment\n"
            f"- Clean white background, no decorative elements, no 3D effects, no shadows\n"
            f"- All text must be horizontal, sans-serif font\n\n"
            f"Output the prompt text (1500-3000 characters). Be specific and detailed."
        )
        figure_prompt_config = self.config.for_stage("figure_prompt")
        try:
            image_prompt = await self._dispatcher.generate(
                figure_prompt_config, FIGURE_PROMPT_SYSTEM, user_prompt
            )
        except Exception as e:
            logger.warning("LLM prompt generation failed for %s: %s", fig_key, e)
            image_prompt = f"{template}\n\nContext: {description}"
        image_prompt = image_prompt.strip()

        # Truncate for safety
        if len(image_prompt) > MAX_IMAGE_PROMPT_LEN:
            truncated = image_prompt[:MAX_IMAGE_PROMPT_LEN].rsplit(" ", 1)
            image_prompt = truncated[0] if len(truncated) > 1 else image_prompt[:MAX_IMAGE_PROMPT_LEN]
            self.log(f"  {fig_key} prompt truncated to {len(image_prompt)} chars")

        self.log(f"  {fig_key} prompt generated ({len(image_prompt)} chars)")
        self.workspace.write_text(
            f"figures/{filename_stem}_prompt.txt", image_prompt
        )

        # Step 2: Generate image via Gemini with retry loop
        figure_gen_config = self.config.for_stage("figure_gen")
        last_error = ""
        prev_error = ""

        for attempt in range(MAX_IMAGE_RETRIES + 1):
            # Early-exit on repeated identical errors
            if attempt >= 1 and last_error and last_error == prev_error:
                self.log(f"  {fig_key} same image error repeated — skipping to diagnosis")
                break
            prev_error = last_error

            try:
                b64_images = await self._dispatcher.generate_image(
                    figure_gen_config, prompt=image_prompt,
                )
                if b64_images:
                    self.log(f"  {fig_key} image generated on attempt {attempt + 1}")
                    result = await self._save_figure_files(
                        fig_key, filename_stem,
                        clean_caption,
                        base64.b64decode(b64_images[0]),
                    )
                    result["generation_prompt"] = image_prompt
                    return result
                last_error = "API returned no image data"
            except Exception as exc:
                last_error = str(exc)

            self.log(f"  {fig_key} attempt {attempt + 1}/{MAX_IMAGE_RETRIES + 1} failed: {last_error}")

        # Step 3: All retries failed — LLM diagnoses and optimizes the prompt
        self.log(f"  {fig_key} all {MAX_IMAGE_RETRIES + 1} attempts failed, running LLM diagnosis")
        optimized_prompt = await self._diagnose_and_optimize_prompt(
            fig_key, image_prompt, last_error, description,
        )

        if optimized_prompt:
            self.workspace.write_text(
                f"figures/{filename_stem}_prompt_optimized.txt", optimized_prompt
            )
            # Step 4: Retry with optimized prompt (2 more attempts)
            for opt_attempt in range(2):
                try:
                    b64_images = await self._dispatcher.generate_image(
                        figure_gen_config, prompt=optimized_prompt,
                    )
                    if b64_images:
                        self.log(f"  {fig_key} succeeded with optimized prompt (attempt {opt_attempt + 1})")
                        result = await self._save_figure_files(
                            fig_key, filename_stem,
                            clean_caption,
                            base64.b64decode(b64_images[0]),
                        )
                        result["generation_prompt"] = optimized_prompt
                        return result
                except Exception as exc:
                    self.log(f"  {fig_key} optimized attempt {opt_attempt + 1} failed: {exc}")

        # Step 5: Final fallback — code-generated placeholder
        self.log(f"  {fig_key} all AI generation attempts exhausted, using fallback")
        result = await self._generate_fallback_chart(fig_key, filename_stem, clean_caption)
        result["generation_prompt"] = image_prompt
        return result

    async def _diagnose_and_optimize_prompt(
        self,
        fig_key: str,
        original_prompt: str,
        last_error: str,
        figure_description: str,
    ) -> str | None:
        """Use LLM to diagnose why image generation failed and produce a shorter, optimized prompt."""
        diagnosis_system = (
            "You are an expert at debugging AI image generation failures. "
            "Analyze the error and the original prompt, then produce an optimized "
            "prompt that avoids the issue while preserving the figure's scientific content."
        )
        diagnosis_user = (
            f"The Gemini image generation API failed for figure '{fig_key}'.\n\n"
            f"Error: {last_error[:500]}\n\n"
            f"Original prompt ({len(original_prompt)} characters):\n"
            f"---\n{original_prompt[:3000]}\n---\n\n"
            f"Figure purpose: {figure_description[:500]}\n\n"
            f"Common failure causes:\n"
            f"1. Prompt too long or complex (Gemini image gen works best with <1500 chars)\n"
            f"2. Too many specific layout instructions (pixel sizes, hex colors, font specs)\n"
            f"3. Requesting text rendering that the model can't do well\n"
            f"4. Content that triggers safety filters\n\n"
            f"Tasks:\n"
            f"1. Diagnose the most likely cause of failure\n"
            f"2. Write an OPTIMIZED prompt (800-1200 characters) that:\n"
            f"   - Preserves the SAME figure content (method names, components, data flow)\n"
            f"   - Uses simpler, more descriptive language\n"
            f"   - Removes pixel-level instructions, specific hex codes, font sizes\n"
            f"   - Describes WHAT to draw, not HOW to render it\n"
            f"   - Keeps it as a clean, 2D scientific diagram\n\n"
            f"Return JSON:\n"
            f'{{"diagnosis": "brief explanation", "optimized_prompt": "the new prompt"}}'
        )

        figure_prompt_config = self.config.for_stage("figure_prompt")
        try:
            result = await self.generate_json(
                diagnosis_system, diagnosis_user, stage_override=figure_prompt_config,
            )
            diagnosis = result.get("diagnosis", "unknown")
            optimized = result.get("optimized_prompt", "")
            self.log(f"  {fig_key} diagnosis: {diagnosis}")

            if not optimized:
                return None

            # Enforce length limit on optimized prompt
            if len(optimized) > MAX_OPTIMIZED_PROMPT_LEN:
                optimized = optimized[:MAX_OPTIMIZED_PROMPT_LEN].rsplit(" ", 1)[0]

            self.log(f"  {fig_key} optimized prompt: {len(optimized)} chars")
            return optimized
        except Exception as exc:
            self.log(f"  {fig_key} prompt diagnosis failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # Code-generated charts (executed in subprocess)
    # -----------------------------------------------------------------------
