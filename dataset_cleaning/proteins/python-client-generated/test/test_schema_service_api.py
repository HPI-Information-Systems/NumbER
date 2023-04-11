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
from swagger_client.api.schema_service_api import SchemaServiceApi  # noqa: E501
from swagger_client.rest import ApiException


class TestSchemaServiceApi(unittest.TestCase):
    """SchemaServiceApi unit test stubs"""

    def setUp(self):
        self.api = SchemaServiceApi()  # noqa: E501

    def tearDown(self):
        pass

    def test_get_assembly_metadata(self):
        """Test case for get_assembly_metadata

        Get assembly schema.  # noqa: E501
        """
        pass

    def test_get_branched_entity_instance_metadata(self):
        """Test case for get_branched_entity_instance_metadata

        Get branched instance schema.  # noqa: E501
        """
        pass

    def test_get_branched_entity_metadata(self):
        """Test case for get_branched_entity_metadata

        Get branched entity schema.  # noqa: E501
        """
        pass

    def test_get_chem_comp_metadata(self):
        """Test case for get_chem_comp_metadata

        Get chemical component/BIRD schema.  # noqa: E501
        """
        pass

    def test_get_drugbank_metadata(self):
        """Test case for get_drugbank_metadata

        Get DrugBank (integrated data) schema.  # noqa: E501
        """
        pass

    def test_get_entry_metadata(self):
        """Test case for get_entry_metadata

        Get entry schema.  # noqa: E501
        """
        pass

    def test_get_non_polymer_entity_instance_metadata(self):
        """Test case for get_non_polymer_entity_instance_metadata

        Get non-polymer instance schema.  # noqa: E501
        """
        pass

    def test_get_non_polymer_entity_metadata(self):
        """Test case for get_non_polymer_entity_metadata

        Get non-polymer entity schema.  # noqa: E501
        """
        pass

    def test_get_polymer_entity_instance_metadata(self):
        """Test case for get_polymer_entity_instance_metadata

        Get polymer instance schema.  # noqa: E501
        """
        pass

    def test_get_polymer_entity_metadata(self):
        """Test case for get_polymer_entity_metadata

        Get polymer entity schema.  # noqa: E501
        """
        pass

    def test_get_pubmed_metadata(self):
        """Test case for get_pubmed_metadata

        Get PubMed (integrated data) schema.  # noqa: E501
        """
        pass

    def test_get_uniprot_metadata(self):
        """Test case for get_uniprot_metadata

        Get UniProt (integrated data) schema.  # noqa: E501
        """
        pass


if __name__ == '__main__':
    unittest.main()
