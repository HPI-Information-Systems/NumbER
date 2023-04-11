# coding: utf-8

"""
    RCSB RESTful API

    Provides programmatic access to information and annotations stored in the Protein Data Bank. <br>Models are generated from JSON schema version: <b>1.40.0</b>. <br>API services deployed on: Sun, 2 Apr 2023 21:44:46 -0700  # noqa: E501

    OpenAPI spec version: 1.40.0
    Contact: info@rcsb.org
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class RcsbUniprotProteinSourceOrganism(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'scientific_name': 'str',
        'taxonomy_id': 'int',
        'provenance_code': 'str'
    }

    attribute_map = {
        'scientific_name': 'scientific_name',
        'taxonomy_id': 'taxonomy_id',
        'provenance_code': 'provenance_code'
    }

    def __init__(self, scientific_name=None, taxonomy_id=None, provenance_code=None):  # noqa: E501
        """RcsbUniprotProteinSourceOrganism - a model defined in Swagger"""  # noqa: E501
        self._scientific_name = None
        self._taxonomy_id = None
        self._provenance_code = None
        self.discriminator = None
        self.scientific_name = scientific_name
        if taxonomy_id is not None:
            self.taxonomy_id = taxonomy_id
        self.provenance_code = provenance_code

    @property
    def scientific_name(self):
        """Gets the scientific_name of this RcsbUniprotProteinSourceOrganism.  # noqa: E501

        The scientific name of the organism in which a protein occurs.  # noqa: E501

        :return: The scientific_name of this RcsbUniprotProteinSourceOrganism.  # noqa: E501
        :rtype: str
        """
        return self._scientific_name

    @scientific_name.setter
    def scientific_name(self, scientific_name):
        """Sets the scientific_name of this RcsbUniprotProteinSourceOrganism.

        The scientific name of the organism in which a protein occurs.  # noqa: E501

        :param scientific_name: The scientific_name of this RcsbUniprotProteinSourceOrganism.  # noqa: E501
        :type: str
        """
        if scientific_name is None:
            raise ValueError("Invalid value for `scientific_name`, must not be `None`")  # noqa: E501

        self._scientific_name = scientific_name

    @property
    def taxonomy_id(self):
        """Gets the taxonomy_id of this RcsbUniprotProteinSourceOrganism.  # noqa: E501

        NCBI Taxonomy identifier for the organism in which a protein occurs.  # noqa: E501

        :return: The taxonomy_id of this RcsbUniprotProteinSourceOrganism.  # noqa: E501
        :rtype: int
        """
        return self._taxonomy_id

    @taxonomy_id.setter
    def taxonomy_id(self, taxonomy_id):
        """Sets the taxonomy_id of this RcsbUniprotProteinSourceOrganism.

        NCBI Taxonomy identifier for the organism in which a protein occurs.  # noqa: E501

        :param taxonomy_id: The taxonomy_id of this RcsbUniprotProteinSourceOrganism.  # noqa: E501
        :type: int
        """

        self._taxonomy_id = taxonomy_id

    @property
    def provenance_code(self):
        """Gets the provenance_code of this RcsbUniprotProteinSourceOrganism.  # noqa: E501

        Historical record of the data attribute.  # noqa: E501

        :return: The provenance_code of this RcsbUniprotProteinSourceOrganism.  # noqa: E501
        :rtype: str
        """
        return self._provenance_code

    @provenance_code.setter
    def provenance_code(self, provenance_code):
        """Sets the provenance_code of this RcsbUniprotProteinSourceOrganism.

        Historical record of the data attribute.  # noqa: E501

        :param provenance_code: The provenance_code of this RcsbUniprotProteinSourceOrganism.  # noqa: E501
        :type: str
        """
        if provenance_code is None:
            raise ValueError("Invalid value for `provenance_code`, must not be `None`")  # noqa: E501

        self._provenance_code = provenance_code

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(RcsbUniprotProteinSourceOrganism, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, RcsbUniprotProteinSourceOrganism):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
