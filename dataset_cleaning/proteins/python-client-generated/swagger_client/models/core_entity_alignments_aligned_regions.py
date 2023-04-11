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

class CoreEntityAlignmentsAlignedRegions(object):
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
        'target_begin': 'int',
        'query_begin': 'int',
        'length': 'int'
    }

    attribute_map = {
        'target_begin': 'target_begin',
        'query_begin': 'query_begin',
        'length': 'length'
    }

    def __init__(self, target_begin=None, query_begin=None, length=None):  # noqa: E501
        """CoreEntityAlignmentsAlignedRegions - a model defined in Swagger"""  # noqa: E501
        self._target_begin = None
        self._query_begin = None
        self._length = None
        self.discriminator = None
        self.target_begin = target_begin
        self.query_begin = query_begin
        self.length = length

    @property
    def target_begin(self):
        """Gets the target_begin of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501

        NCBI sequence start position  # noqa: E501

        :return: The target_begin of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501
        :rtype: int
        """
        return self._target_begin

    @target_begin.setter
    def target_begin(self, target_begin):
        """Sets the target_begin of this CoreEntityAlignmentsAlignedRegions.

        NCBI sequence start position  # noqa: E501

        :param target_begin: The target_begin of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501
        :type: int
        """
        if target_begin is None:
            raise ValueError("Invalid value for `target_begin`, must not be `None`")  # noqa: E501

        self._target_begin = target_begin

    @property
    def query_begin(self):
        """Gets the query_begin of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501

        Entity seqeunce start position  # noqa: E501

        :return: The query_begin of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501
        :rtype: int
        """
        return self._query_begin

    @query_begin.setter
    def query_begin(self, query_begin):
        """Sets the query_begin of this CoreEntityAlignmentsAlignedRegions.

        Entity seqeunce start position  # noqa: E501

        :param query_begin: The query_begin of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501
        :type: int
        """
        if query_begin is None:
            raise ValueError("Invalid value for `query_begin`, must not be `None`")  # noqa: E501

        self._query_begin = query_begin

    @property
    def length(self):
        """Gets the length of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501

        Aligned region length  # noqa: E501

        :return: The length of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501
        :rtype: int
        """
        return self._length

    @length.setter
    def length(self, length):
        """Sets the length of this CoreEntityAlignmentsAlignedRegions.

        Aligned region length  # noqa: E501

        :param length: The length of this CoreEntityAlignmentsAlignedRegions.  # noqa: E501
        :type: int
        """
        if length is None:
            raise ValueError("Invalid value for `length`, must not be `None`")  # noqa: E501

        self._length = length

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
        if issubclass(CoreEntityAlignmentsAlignedRegions, dict):
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
        if not isinstance(other, CoreEntityAlignmentsAlignedRegions):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
