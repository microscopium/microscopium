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


def test_ix_semantic_filename2():
    test_fn = "./BBBC022_v1_images_20585w1/IXMtest_L09_s3_w1538679C9-F03A-" \
              "4656-9A57-0D4A440C1C62.tif"
    expected = coll.OrderedDict([('directory', './BBBC022_v1_images_20585w1'),
                                 ('prefix', 'IXMtest'),
                                 ('plate', 20585),
                                 ('well', 'L09'),
                                 ('field', 2),
                                 ('channel', 0),
                                 ('suffix', 'tif')])
    assert image_xpress.ix_semantic_filename(test_fn) == expected

