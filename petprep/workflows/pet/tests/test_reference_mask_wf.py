from .... import data
from ...tests import mock_config
from ..reference_mask import init_pet_refmask_wf


def test_refmask_morph_nodes():
    with mock_config():
        cfg = str(data.load('reference_mask/config.json'))
        wf = init_pet_refmask_wf(
            segmentation='gtm',
            ref_mask_name='cerebellum',
            config_path=cfg,
        )
        node_names = [n.name for n in wf._get_all_nodes()]
        assert 'make_morphtsv' in node_names
        assert 'ds_morphtsv' in node_names
        extract = wf.get_node('extract_refregion')
        make_stats = wf.get_node('make_morphtsv')
        edge = wf._graph.get_edge_data(extract, make_stats)
        assert ('refmask_file', 'mask_file') in edge['connect']
