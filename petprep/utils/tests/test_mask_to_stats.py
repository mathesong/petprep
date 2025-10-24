import nibabel as nb
import numpy as np
import pandas as pd

from petprep.utils.reference_mask import mask_to_stats


def test_mask_to_stats(tmp_path):
    data = np.ones((2, 2, 2), dtype=np.uint8)
    img = nb.Nifti1Image(data, np.eye(4))
    mask_file = tmp_path / 'mask.nii.gz'
    img.to_filename(mask_file)

    out = mask_to_stats(str(mask_file), 'testmask')
    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['index', 'name', 'volume-mm3']
    assert df.iloc[0]['index'] == 1
    assert df.iloc[0]['name'] == 'testmask'
    expected_vol = data.sum() * np.prod(img.header.get_zooms())
    assert df.iloc[0]['volume-mm3'] == expected_vol
