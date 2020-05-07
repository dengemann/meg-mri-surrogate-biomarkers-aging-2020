"""Setup the main package."""


def configuration(parent_package='', top_path=None):
    """Configure the package."""
    from numpy.distutils.misc_util import Configuration

    config = Configuration('camcan', parent_package, top_path)

    config.add_subpackage('datasets')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('datasets/tests/data')
    config.add_subpackage('datasets/tests/data/sub-0')
    config.add_subpackage('datasets/tests/data/sub-1')
    config.add_subpackage('preprocessing')
    config.add_subpackage('preprocessing/tests')
    config.add_subpackage('preprocessing/tests/data')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)
