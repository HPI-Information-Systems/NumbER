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

class RcsbEntitySourceOrganismRcsbGeneName(object):
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
        'provenance_source': 'str',
        'value': 'str'
    }

    attribute_map = {
        'provenance_source': 'provenance_source',
        'value': 'value'
    }

    def __init__(self, provenance_source=None, value=None):  # noqa: E501
        """RcsbEntitySourceOrganismRcsbGeneName - a model defined in Swagger"""  # noqa: E501
        self._provenance_source = None
        self._value = None
        self.discriminator = None
        if provenance_source is not None:
            self.provenance_source = provenance_source
        if value is not None:
            self.value = value

    @property
    def provenance_source(self):
        """Gets the provenance_source of this RcsbEntitySourceOrganismRcsbGeneName.  # noqa: E501

        A code indicating the provenance of the source organism details for the entity  # noqa: E501

        :return: The provenance_source of this RcsbEntitySourceOrganismRcsbGeneName.  # noqa: E501
        :rtype: str
        """
        return self._provenance_source

    @provenance_source.setter
    def provenance_source(self, provenance_source):
        """Sets the provenance_source of this RcsbEntitySourceOrganismRcsbGeneName.

        A code indicating the provenance of the source organism details for the entity  # noqa: E501

        :param provenance_source: The provenance_source of this RcsbEntitySourceOrganismRcsbGeneName.  # noqa: E501
        :type: str
        """
        allowed_values = ["PDB Primary Data", "UniProt"]  # noqa: E501
        # if provenance_source not in allowed_values:
        #     raise ValueError(
        #         "Invalid value for `provenance_source` ({0}), must be one of {1}"  # noqa: E501
        #         .format(provenance_source, allowed_values)
        #     )

        self._provenance_source = provenance_source

    @property
    def value(self):
        """Gets the value of this RcsbEntitySourceOrganismRcsbGeneName.  # noqa: E501

        Gene name.  # noqa: E501

        :return: The value of this RcsbEntitySourceOrganismRcsbGeneName.  # noqa: E501
        :rtype: str
        """
        return self._value

    @value.setter
    def value(self, value):
        """Sets the value of this RcsbEntitySourceOrganismRcsbGeneName.

        Gene name.  # noqa: E501

        :param value: The value of this RcsbEntitySourceOrganismRcsbGeneName.  # noqa: E501
        :type: str
        """

        self._value = value

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
        if issubclass(RcsbEntitySourceOrganismRcsbGeneName, dict):
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
        if not isinstance(other, RcsbEntitySourceOrganismRcsbGeneName):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
