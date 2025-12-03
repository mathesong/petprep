# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Registration workflows
++++++++++++++++++++++

.. autofunction:: init_pet_reg_wf

"""

import typing as ty

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

AffineDOF = ty.Literal[6, 9, 12]


def _get_first(in_list):
    """Extract first element from a list (for ANTs transform output)."""
    return in_list[0]


def _fsl2itk(fsl_matrix, source_file, reference_file):
    """
    Convert FSL FLIRT matrix to ITK format.

    Parameters
    ----------
    fsl_matrix : str
        Path to FSL .mat file
    source_file : str
        Source image (the image that was transformed)
    reference_file : str
        Reference image (the target space)

    Returns
    -------
    str
        Path to ITK transform file
    """
    import os
    import numpy as np
    import nibabel as nb
    from pathlib import Path

    # Load images to get headers
    src_img = nb.load(source_file)
    ref_img = nb.load(reference_file)

    # Load FSL matrix
    fsl_xfm = np.loadtxt(fsl_matrix)

    # Get voxel-to-RAS matrices
    src_vox2ras = src_img.affine
    ref_vox2ras = ref_img.affine

    # Convert FSL (voxel-to-voxel) to RAS-to-RAS
    # FSL: vox_ref = M * vox_src
    # ITK: ras_ref = T * ras_src
    # T = ref_vox2ras * M * inv(src_vox2ras)
    itk_xfm = ref_vox2ras @ fsl_xfm @ np.linalg.inv(src_vox2ras)

    # ITK uses the inverse convention (from reference to source)
    itk_xfm = np.linalg.inv(itk_xfm)

    # Write ITK format transform (simple affine text format)
    out_file = os.path.join(os.getcwd(), 'fsl2itk_transform.txt')

    # Write in ITK format (12 parameters: 9 rotation/scale + 3 translation)
    with open(out_file, 'w') as f:
        f.write('#Insight Transform File V1.0\n')
        f.write('#Transform 0\n')
        f.write('Transform: AffineTransform_double_3_3\n')
        f.write('Parameters: ')
        # ITK order: first 9 are the matrix (row-major), last 3 are translation
        params = []
        for i in range(3):
            for j in range(3):
                params.append(str(itk_xfm[i, j]))
        for i in range(3):
            params.append(str(itk_xfm[i, 3]))
        f.write(' '.join(params) + '\n')
        f.write('FixedParameters: 0 0 0\n')

    return out_file


def init_pet_reg_wf(
    *,
    pet2anat_dof: AffineDOF,
    mem_gb: float,
    omp_nthreads: int,
    pet2anat_method: str = 'mri_coreg',
    name: str = 'pet_reg_wf',
    sloppy: bool = False,
):
    """
    Build a workflow to run same-subject, PET-to-anat image-registration.

    Calculates the registration between a reference PET image and anat-space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.pet.registration import init_pet_reg_wf
            wf = init_pet_reg_wf(
                mem_gb=3,
                omp_nthreads=1,
                pet2anat_dof=6,
            )

    Parameters
    ----------
    pet2anat_dof : 6, 9 or 12
        Degrees-of-freedom for PET-anatomical registration
    mem_gb : :obj:`float`
        Size of PET file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    pet2anat_method : :obj:`str`
        Method for PET-to-anatomical registration. Options are 'mri_coreg'
        (default FreeSurfer co-registration), 'robust' (uses FreeSurfer
        mri_robust_register with NMI, 6 DoF only), or 'ants' (uses ANTs rigid
        registration, 6 DoF only).
    name : :obj:`str`
        Name of workflow (default: ``pet_reg_wf``)

    Inputs
    ------
    ref_pet_brain
        Reference image to which PET series is aligned
        If ``fieldwarp == True``, ``ref_pet_brain`` should be unwarped
    anat_preproc
        Preprocessed anatomical image
    anat_mask
        Brainmask for anatomical image

    Outputs
    -------
    itk_pet_to_anat
        Affine transform from ``ref_pet_brain`` to anatomical space (ITK format)
    itk_anat_to_pet
        Affine transform from anatomical space to PET space (ITK format)

    """
    from nipype.interfaces.ants import Registration
    from nipype.interfaces.freesurfer import MRICoreg, RobustRegister
    from nipype.interfaces.fsl import FLIRT, RobustFOV
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['ref_pet_brain', 'anat_preproc', 'anat_mask']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['itk_pet_to_t1', 'itk_t1_to_pet']),
        name='outputnode',
    )

    mask_brain = pe.Node(ApplyMask(), name='mask_brain')
    crop_anat_mask = pe.Node(
        FLIRT(apply_xfm=True, interp='nearestneighbour', output_type='NIFTI_GZ'),
        name='crop_anat_mask',
    )
    robust_fov = pe.Node(RobustFOV(output_type='NIFTI_GZ'), name='robust_fov')

    # Convert robustFOV transform (FSL matrix) to ITK format
    # robustFOV outputs: original_T1w -> cropped_T1w
    fsl2itk = pe.Node(
        niu.Function(
            input_names=['fsl_matrix', 'source_file', 'reference_file'],
            output_names=['out_file'],
            function=_fsl2itk,
        ),
        name='fsl2itk',
    )

    # Invert robustFOV transform to get: cropped_T1w -> original_T1w
    invert_crop_xfm = pe.Node(
        ConcatenateXFMs(inverse=True),
        name='invert_crop_xfm',
    )

    if pet2anat_method == 'ants':
        coreg = pe.Node(
            Registration(
                dimension=3,
                float=True,
                output_transform_prefix='pet2anat_',
                output_warped_image='pet2anat_Warped.nii.gz',
                transforms=['Rigid'],
                transform_parameters=[(0.1,)],
                metric=['MI'],
                metric_weight=[1],
                radius_or_number_of_bins=[32],
                sampling_strategy=['Regular'],
                sampling_percentage=[0.25],
                number_of_iterations=[[1000, 500, 250]],
                convergence_threshold=[1e-6],
                convergence_window_size=[10],
                shrink_factors=[[4, 2, 1]],
                smoothing_sigmas=[[2, 1, 0]],
                sigma_units=['vox'],
                use_histogram_matching=False,
                initial_moving_transform_com=1,
                winsorize_lower_quantile=0.001,
                winsorize_upper_quantile=0.999,
            ),
            name='ants_registration',
            n_procs=omp_nthreads,
            mem_gb=mem_gb * 2,
        )
        coreg_target = 'fixed_image'
        coreg_mask = 'fixed_image_masks'
        coreg_moving = 'moving_image'
        coreg_output = 'forward_transforms'
        coreg_output_is_list = True

    elif pet2anat_method == 'robust':
        coreg = pe.Node(
            RobustRegister(
                auto_sens=False,
                est_int_scale=False,
                init_orient=True,
                args='--cost NMI',
                max_iterations=10,
                high_iterations=20,
                iteration_thresh=0.01,
            ),
            name='mri_robust_register',
            n_procs=omp_nthreads,
            mem_gb=5,
        )
        coreg_target = 'target_file'
        coreg_moving = 'source_file'
        coreg_output = 'out_reg_file'
        coreg_output_is_list = False

    else:  # mri_coreg (default)
        coreg = pe.Node(
            MRICoreg(dof=pet2anat_dof, sep=[4], ftol=0.0001, linmintol=0.01),
            name='mri_coreg',
            n_procs=omp_nthreads,
            mem_gb=5,
        )
        coreg_target = 'reference_file'
        coreg_moving = 'source_file'
        coreg_output = 'out_lta_file'
        coreg_output_is_list = False

    # Merge transforms into a list: [inv(robustFOV), registration] = [cropped->original, PET->cropped]
    # ITK applies right-to-left: PET->cropped first, then cropped->original = PET->original
    merge_xfms = pe.Node(niu.Merge(2), name='merge_xfms')

    # Concatenate the transforms
    concat_xfm = pe.Node(ConcatenateXFMs(inverse=False), name='concat_xfm')

    # Get inverse for original_T1w -> PET
    convert_xfm = pe.Node(ConcatenateXFMs(inverse=True), name='convert_xfm')

    # Build connections dynamically based on output type
    if coreg_output_is_list:
        # ANTs outputs a list of transforms; take the first (and only) one
        # ANTs gets unmasked T1W + separate mask (not pre-masked image)
        # ANTs computes: PET -> cropped_T1w
        # Need to concatenate with inverse of robustFOV to get: PET -> original_T1w
        connections = [
            (robust_fov, mask_brain, [('out_roi', 'in_file')]),
            (crop_anat_mask, mask_brain, [('out_file', 'in_mask')]),
            (inputnode, coreg, [('ref_pet_brain', coreg_moving)]),
            (
                robust_fov,
                coreg,
                [
                    ('out_roi', coreg_target),
                ],
            ),
            (crop_anat_mask, coreg, [('out_file', coreg_mask)]),
            # Convert FSL robustFOV transform (original->cropped) to ITK format
            (robust_fov, fsl2itk, [('out_transform', 'fsl_matrix')]),
            (inputnode, fsl2itk, [('anat_preproc', 'source_file')]),  # original T1w
            (robust_fov, fsl2itk, [('out_roi', 'reference_file')]),  # cropped T1w
            # Invert robustFOV: original->cropped becomes cropped->original
            (fsl2itk, invert_crop_xfm, [('out_file', 'in_xfms')]),
            # Merge transforms into list: [cropped->original, PET->cropped]
            (invert_crop_xfm, merge_xfms, [('out_xfm', 'in1')]),  # cropped->original
            (coreg, merge_xfms, [((coreg_output, _get_first), 'in2')]),  # PET->cropped
            # Concatenate the merged transforms
            (merge_xfms, concat_xfm, [('out', 'in_xfms')]),
            # Result: PET->original. Get forward and inverse
            (concat_xfm, convert_xfm, [('out_xfm', 'in_xfms')]),
            (
                convert_xfm,
                outputnode,
                [
                    ('out_xfm', 'itk_pet_to_t1'),  # PET -> original_T1w
                    ('out_inv', 'itk_t1_to_pet'),  # original_T1w -> PET
                ],
            ),
        ]
    else:
        # mri_coreg and robust output single transform file
        # They also register to cropped+masked brain, so need concatenation
        connections = [
            (robust_fov, mask_brain, [('out_roi', 'in_file')]),
            (crop_anat_mask, mask_brain, [('out_file', 'in_mask')]),
            (inputnode, coreg, [('ref_pet_brain', coreg_moving)]),
            (mask_brain, coreg, [('out_file', coreg_target)]),  # Uses cropped brain
            # Convert FSL robustFOV transform (original->cropped) to ITK format
            (robust_fov, fsl2itk, [('out_transform', 'fsl_matrix')]),
            (inputnode, fsl2itk, [('anat_preproc', 'source_file')]),  # original T1w
            (robust_fov, fsl2itk, [('out_roi', 'reference_file')]),  # cropped T1w
            # Invert robustFOV: original->cropped becomes cropped->original
            (fsl2itk, invert_crop_xfm, [('out_file', 'in_xfms')]),
            # Merge transforms into list: [cropped->original, PET->cropped]
            (invert_crop_xfm, merge_xfms, [('out_xfm', 'in1')]),  # cropped->original
            (coreg, merge_xfms, [(coreg_output, 'in2')]),  # PET->cropped
            # Concatenate the merged transforms
            (merge_xfms, concat_xfm, [('out', 'in_xfms')]),
            # Result: PET->original. Get forward and inverse
            (concat_xfm, convert_xfm, [('out_xfm', 'in_xfms')]),
            (
                convert_xfm,
                outputnode,
                [
                    ('out_xfm', 'itk_pet_to_t1'),  # PET -> original_T1w
                    ('out_inv', 'itk_t1_to_pet'),  # original_T1w -> PET
                ],
            ),
        ]

    workflow.connect(
        [
            (inputnode, robust_fov, [('anat_preproc', 'in_file')]),
            (inputnode, crop_anat_mask, [('anat_mask', 'in_file')]),
            (robust_fov, crop_anat_mask, [('out_roi', 'reference'), ('out_transform', 'in_matrix_file')]),
        ]
        + connections
    )  # fmt:skip

    return workflow
