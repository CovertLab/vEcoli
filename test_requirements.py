import os
from pkg_resources import parse_requirements, Requirement

from setup import INSTALL_REQUIRES


REQUIREMENTS_PATH = os.path.join(
    os.path.dirname(__file__), 'requirements.txt')


def test_requirements():
    '''Check that the requirements in requirements.txt are compatible
    with those specified in setup.py. Note that this does not guarantee
    that the requirements in setup.py are correct--it just does a
    sanity-check to ensure they don't conflict with requirements.txt.'''
    with open(REQUIREMENTS_PATH, 'r') as f:
        txt_requirements = parse_requirements(f)
        txt_requirements_map = {
            req.key: req
            for req in txt_requirements
        }

    for setup_req_str in INSTALL_REQUIRES:
        setup_req = Requirement.parse(setup_req_str)
        txt_req_specs = txt_requirements_map[setup_req.key].specs
        assert len(txt_req_specs) == 1
        txt_req_op, txt_req_version = txt_req_specs[0]
        # The requirements.txt file should only have `==` requirements.
        assert txt_req_op == '=='
        # Check that the requirement in requirements.txt is compatible
        # with the requirement in setup.py.
        assert txt_req_version in setup_req


if __name__ == '__main__':
    test_requirements()
