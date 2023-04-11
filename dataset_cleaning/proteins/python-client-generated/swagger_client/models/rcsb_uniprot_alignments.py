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

class RcsbUniprotAlignments(object):
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
        'core_entity_alignments': 'list[RcsbUniprotAlignmentsCoreEntityAlignments]'
    }

    attribute_map = {
        'core_entity_alignments': 'core_entity_alignments'
    }

    def __init__(self, core_entity_alignments=None):  # noqa: E501
        """RcsbUniprotAlignments - a model defined in Swagger"""  # noqa: E501
        self._core_entity_alignments = None
        self.discriminator = None
        if core_entity_alignments is not None:
            self.core_entity_alignments = core_entity_alignments

    @property
    def core_entity_alignments(self):
        """Gets the core_entity_alignments of this RcsbUniprotAlignments.  # noqa: E501


        :return: The core_entity_alignments of this RcsbUniprotAlignments.  # noqa: E501
        :rtype: list[RcsbUniprotAlignmentsCoreEntityAlignments]
        """
        return self._core_entity_alignments

    @core_entity_alignments.setter
    def core_entity_alignments(self, core_entity_alignments):
        """Sets the core_entity_alignments of this RcsbUniprotAlignments.


        :param core_entity_alignments: The core_entity_alignments of this RcsbUniprotAlignments.  # noqa: E501
        :type: list[RcsbUniprotAlignmentsCoreEntityAlignments]
        """

        self._core_entity_alignments = core_entity_alignments

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
        if issubclass(RcsbUniprotAlignments, dict):
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
        if not isinstance(other, RcsbUniprotAlignments):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
