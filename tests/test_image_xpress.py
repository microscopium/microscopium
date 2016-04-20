from microscopium.screens import image_xpress
import collections as coll


def test_ix_semantic_filename():
    test_fn = "./Week1_22123/G10_s2_w11C3B9BCC-E48F-4C2F-9D31-8F46D8B5B972.tif"
    expected = coll.OrderedDict([('directory', './Week1_22123'),
                            ('prefix', ''),
                            ('plate', 22123),
                            ('well', 'G10'),
                            ('field', 1),
                            ('channel', 0),
                            ('suffix', 'tif')])

    assert image_xpress.ix_semantic_filename(test_fn) == expected

