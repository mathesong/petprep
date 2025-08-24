from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe

from ...interfaces import ExtractRefTAC


def resample_pet_to_mask(pet_file, mask_file):
    import os

    from nilearn.image import resample_to_img

    resampled = resample_to_img(pet_file, mask_file, interpolation='continuous')
    out_file = os.path.abspath('pet_resampled.nii.gz')
    resampled.to_filename(out_file)
    return out_file


def init_pet_ref_tacs_wf(*, name: str = 'pet_ref_tacs_wf') -> pe.Workflow:
    """Extract reference region time activity curve."""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_anat', 'mask_file', 'metadata', 'ref_mask_name']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['tacs']), name='outputnode')

    resample_pet = pe.Node(
        Function(
            input_names=['pet_file', 'mask_file'],
            output_names=['resampled_pet'],
            function=resample_pet_to_mask,
        ),
        name='resample_pet',
    )

    tac = pe.Node(ExtractRefTAC(), name='tac')

    workflow.connect(
        [
            (
                inputnode,
                resample_pet,
                [('pet_anat', 'pet_file'), ('mask_file', 'mask_file')],
            ),
            (resample_pet, tac, [('resampled_pet', 'in_file')]),
            (
                inputnode,
                tac,
                [
                    ('mask_file', 'mask_file'),
                    ('metadata', 'metadata'),
                    ('ref_mask_name', 'ref_mask_name'),
                ],
            ),
            (tac, outputnode, [('out_file', 'tacs')]),
        ]
    )

    return workflow


__all__ = ('init_pet_ref_tacs_wf',)
