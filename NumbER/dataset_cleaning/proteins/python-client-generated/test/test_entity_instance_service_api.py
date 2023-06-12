# coding: utf-8

"""
    RCSB RESTful API

    Provides programmatic access to information and annotations stored in the Protein Data Bank. <br>Models are generated from JSON schema version: <b>1.40.0</b>. <br>API services deployed on: Sun, 2 Apr 2023 21:44:46 -0700  # noqa: E501

    OpenAPI spec version: 1.40.0
    Contact: info@rcsb.org
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

from __future__ import absolute_import

import unittest

import swagger_client
from swagger_client.api.entity_instance_service_api import EntityInstanceServiceApi  # noqa: E501
from swagger_client.rest import ApiException


class TestEntityInstanceServiceApi(unittest.TestCase):
    """EntityInstanceServiceApi unit test stubs"""

    def setUp(self):
        self.api = EntityInstanceServiceApi()  # noqa: E501

    def tearDown(self):
        pass

    def test_get_branched_entity_instance_by_id(self):
        """Test case for get_branched_entity_instance_by_id

        Get branched entity instance description by ENTRY ID and ASYM ID.  # noqa: E501
        """
        pass

    def test_get_non_polymer_entity_instance_by_id(self):
        """Test case for get_non_polymer_entity_instance_by_id

        Get non-polymer entity instance description by ENTRY ID and ASYM ID (label_asym_id).  # noqa: E501
        """
        pass

    def test_get_polymer_entity_instance_by_id(self):
        """Test case for get_polymer_entity_instance_by_id

        Get polymer entity instance (a.k.a chain) data by ENTRY ID and ASYM ID (label_asym_id).  # noqa: E501
        """
        pass


if __name__ == '__main__':
    unittest.main()