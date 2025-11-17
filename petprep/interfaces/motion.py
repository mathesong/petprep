# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Reportlets illustrating motion correction."""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from imageio import v2 as imageio
from nilearn import image
from nilearn.plotting import plot_epi
from nilearn.plotting.find_cuts import find_xyz_cut_coords
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class MotionPlotInputSpec(BaseInterfaceInputSpec):
    original_pet = File(
        exists=True,
        mandatory=True,
        desc='Original (uncorrected) PET series in native PET space',
    )
    corrected_pet = File(
        exists=True,
        mandatory=True,
        desc=(
            'Motion-corrected PET series derived by applying the estimated motion '
            'transforms to the original data in native PET space'
        ),
    )
    duration = traits.Float(0.2, usedefault=True, desc='Frame duration for the GIF (seconds)')


class MotionPlotOutputSpec(TraitedSpec):
    svg_file = File(exists=True, desc='Animated before/after motion correction SVG')


class MotionPlot(SimpleInterface):
    """Generate animated visualizations before and after motion correction.

    A single GIF is created using ortho views with consistent cut coordinates
    and color scaling derived from the midpoint frame of each series. The
    per-frame views of the original and motion-corrected series are concatenated
    horizontally, allowing the main PET report to display the animation
    directly.
    """

    input_spec = MotionPlotInputSpec
    output_spec = MotionPlotOutputSpec

    def _run_interface(self, runtime):
        runtime.cwd = Path(runtime.cwd)

        svg_file = runtime.cwd / 'pet_motion_hmc.svg'
        svg_file.parent.mkdir(parents=True, exist_ok=True)

        mid_orig, cut_coords_orig, vmin_orig, vmax_orig = self._compute_display_params(
            self.inputs.original_pet
        )
        _, _, vmin_corr, vmax_corr = self._compute_display_params(self.inputs.corrected_pet)

        svg_file = self._build_animation(
            output_path=svg_file,
            cut_coords_orig=cut_coords_orig,
            cut_coords_corr=cut_coords_orig,
            vmin_orig=vmin_orig,
            vmax_orig=vmax_orig,
            vmin_corr=vmin_corr,
            vmax_corr=vmax_corr,
        )

        self._results['svg_file'] = str(svg_file)

        return runtime

    def _compute_display_params(self, in_file: str):
        img = nib.load(in_file)
        if img.ndim == 3:
            mid_img = img
        else:
            mid_img = image.index_img(in_file, img.shape[-1] // 2)

        data = mid_img.get_fdata().astype(float)
        vmax = float(np.percentile(data.flatten(), 99.9))
        vmin = float(np.percentile(data.flatten(), 80))
        cut_coords = find_xyz_cut_coords(mid_img)

        return mid_img, cut_coords, vmin, vmax

    def _build_animation(
        self,
        *,
        output_path: Path,
        cut_coords_orig: tuple[float, float, float],
        cut_coords_corr: tuple[float, float, float],
        vmin_orig: float,
        vmax_orig: float,
        vmin_corr: float,
        vmax_corr: float,
    ) -> Path:
        orig_img = nib.load(self.inputs.original_pet)
        corr_img = nib.load(self.inputs.corrected_pet)

        n_frames = min(
            orig_img.shape[-1] if orig_img.ndim > 3 else 1,
            corr_img.shape[-1] if corr_img.ndim > 3 else 1,
        )

        with TemporaryDirectory() as tmpdir:
            frames = []
            for idx in range(n_frames):
                orig_png = Path(tmpdir) / f'orig_{idx:04d}.png'
                corr_png = Path(tmpdir) / f'corr_{idx:04d}.png'

                plot_epi(
                    image.index_img(self.inputs.original_pet, idx),
                    colorbar=True,
                    display_mode='ortho',
                    title=f'Before motion correction | Frame {idx+1}',
                    cut_coords=cut_coords_orig,
                    vmin=vmin_orig,
                    vmax=vmax_orig,
                    output_file=str(orig_png),
                )
                plot_epi(
                    image.index_img(self.inputs.corrected_pet, idx),
                    colorbar=True,
                    display_mode='ortho',
                    title=f'After motion correction | Frame {idx+1}',
                    cut_coords=cut_coords_corr,
                    vmin=vmin_corr,
                    vmax=vmax_corr,
                    output_file=str(corr_png),
                )

                orig_arr = np.asarray(imageio.imread(orig_png))
                corr_arr = np.asarray(imageio.imread(corr_png))

                max_height = max(orig_arr.shape[0], corr_arr.shape[0])
                if orig_arr.shape[0] < max_height:
                    pad = max_height - orig_arr.shape[0]
                    orig_arr = np.pad(orig_arr, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=255)
                if corr_arr.shape[0] < max_height:
                    pad = max_height - corr_arr.shape[0]
                    corr_arr = np.pad(corr_arr, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=255)

                combined = np.concatenate([orig_arr, corr_arr], axis=1)
                frames.append(combined.astype(orig_arr.dtype, copy=False))

            width = int(frames[0].shape[1])
            height = int(frames[0].shape[0])
            total_duration = self.inputs.duration * n_frames

            svg_parts = [
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<style>',
                (
                    '.frame {'
                    f' opacity: 0; animation: framefade {total_duration}s infinite;'
                    ' animation-play-state: paused;'
                    '}'
                ),
                '.playing .frame {animation-play-state: running;}',
                '@keyframes framefade {0%, 80% {opacity: 1;} 100% {opacity: 0;}}',
            ]

            for idx in range(n_frames):
                delay = self.inputs.duration * idx
                svg_parts.append(f'.frame-{idx} {{animation-delay: {delay}s;}}')

            svg_parts.append('</style>')

            for idx, frame in enumerate(frames):
                buffer = BytesIO()
                imageio.imwrite(buffer, frame, format='PNG')
                data_uri = b64encode(buffer.getvalue()).decode('ascii')
                svg_parts.append(
                    f'<image class="frame frame-{idx}" '
                    f'width="{width}" height="{height}" x="0" y="0" '
                    f'href="data:image/png;base64,{data_uri}" />'
                )

            svg_parts.extend(
                [
                    '<script>',
                    '(() => {',
                    '  const svg = document.currentScript.parentNode;',
                    "  const frames = svg.querySelectorAll('.frame');",
                    f'  const cycleMs = {total_duration * 1000:.0f};',
                    '  let restartTimer = null;',
                    '  const restart = () => {',
                    '    frames.forEach((frame) => {',
                    "      frame.style.animation = 'none';",
                    '      // Force reflow to restart the CSS animation',
                    '      void frame.getBoundingClientRect();',
                    "      frame.style.animation = '';",
                    '    });',
                    '  };',
                    '  const start = () => {',
                    '    if (restartTimer) {',
                    '      clearInterval(restartTimer);',
                    '    }',
                    '    svg.classList.add("playing");',
                    '    restart();',
                    '    restartTimer = setInterval(restart, cycleMs);',
                    '  };',
                    '  const stop = () => {',
                    '    svg.classList.remove("playing");',
                    '    if (restartTimer) {',
                    '      clearInterval(restartTimer);',
                    '      restartTimer = null;',
                    '    }',
                    '    frames.forEach((frame) => {',
                    "      frame.style.animation = 'none';",
                    '    });',
                    '  };',
                    "  svg.addEventListener('mouseenter', start);",
                    "  svg.addEventListener('mouseleave', stop);",
                    '})();',
                    '</script>',
                    '</svg>',
                ]
            )

            output_path.write_text('\n'.join(svg_parts), encoding='utf-8')

        return output_path


__all__ = ['MotionPlot']
