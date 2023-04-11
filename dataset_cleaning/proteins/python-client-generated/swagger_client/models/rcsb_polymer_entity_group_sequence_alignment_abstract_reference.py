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

class RcsbPolymerEntityGroupSequenceAlignmentAbstractReference(object):
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
        'length': 'int',
        'sequence': 'str'
    }

    attribute_map = {
        'length': 'length',
        'sequence': 'sequence'
    }

    def __init__(self, length=None, sequence=None):  # noqa: E501
        """RcsbPolymerEntityGroupSequenceAlignmentAbstractReference - a model defined in Swagger"""  # noqa: E501
        self._length = None
        self._sequence = None
        self.discriminator = None
        self.length = length
        if sequence is not None:
            self.sequence = sequence

    @property
    def length(self):
        """Gets the length of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.  # noqa: E501

        Abstract reference length  # noqa: E501

        :return: The length of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.  # noqa: E501
        :rtype: int
        """
        return self._length

    @length.setter
    def length(self, length):
        """Sets the length of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.

        Abstract reference length  # noqa: E501

        :param length: The length of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.  # noqa: E501
        :type: int
        """
        if length is None:
            raise ValueError("Invalid value for `length`, must not be `None`")  # noqa: E501

        self._length = length

    @property
    def sequence(self):
        """Gets the sequence of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.  # noqa: E501

        Sequence that represents the abstract reference  # noqa: E501

        :return: The sequence of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.  # noqa: E501
        :rtype: str
        """
        return self._sequence

    @sequence.setter
    def sequence(self, sequence):
        """Sets the sequence of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.

        Sequence that represents the abstract reference  # noqa: E501

        :param sequence: The sequence of this RcsbPolymerEntityGroupSequenceAlignmentAbstractReference.  # noqa: E501
        :type: str
        """

        self._sequence = sequence

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
        if issubclass(RcsbPolymerEntityGroupSequenceAlignmentAbstractReference, dict):
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
        if not isinstance(other, RcsbPolymerEntityGroupSequenceAlignmentAbstractReference):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other