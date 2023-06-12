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

class RcsbStructSymmetryRotationAxes(object):
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
        'start': 'list[float]',
        'end': 'list[float]',
        'order': 'int'
    }

    attribute_map = {
        'start': 'start',
        'end': 'end',
        'order': 'order'
    }

    def __init__(self, start=None, end=None, order=None):  # noqa: E501
        """RcsbStructSymmetryRotationAxes - a model defined in Swagger"""  # noqa: E501
        self._start = None
        self._end = None
        self._order = None
        self.discriminator = None
        self.start = start
        self.end = end
        if order is not None:
            self.order = order

    @property
    def start(self):
        """Gets the start of this RcsbStructSymmetryRotationAxes.  # noqa: E501

        coordinate  # noqa: E501

        :return: The start of this RcsbStructSymmetryRotationAxes.  # noqa: E501
        :rtype: list[float]
        """
        return self._start

    @start.setter
    def start(self, start):
        """Sets the start of this RcsbStructSymmetryRotationAxes.

        coordinate  # noqa: E501

        :param start: The start of this RcsbStructSymmetryRotationAxes.  # noqa: E501
        :type: list[float]
        """
        if start is None:
            raise ValueError("Invalid value for `start`, must not be `None`")  # noqa: E501

        self._start = start

    @property
    def end(self):
        """Gets the end of this RcsbStructSymmetryRotationAxes.  # noqa: E501

        coordinate  # noqa: E501

        :return: The end of this RcsbStructSymmetryRotationAxes.  # noqa: E501
        :rtype: list[float]
        """
        return self._end

    @end.setter
    def end(self, end):
        """Sets the end of this RcsbStructSymmetryRotationAxes.

        coordinate  # noqa: E501

        :param end: The end of this RcsbStructSymmetryRotationAxes.  # noqa: E501
        :type: list[float]
        """
        if end is None:
            raise ValueError("Invalid value for `end`, must not be `None`")  # noqa: E501

        self._end = end

    @property
    def order(self):
        """Gets the order of this RcsbStructSymmetryRotationAxes.  # noqa: E501

        The number of times (order of rotation) that a subunit can be repeated by a rotation operation, being transformed into a new state indistinguishable from its starting state.  # noqa: E501

        :return: The order of this RcsbStructSymmetryRotationAxes.  # noqa: E501
        :rtype: int
        """
        return self._order

    @order.setter
    def order(self, order):
        """Sets the order of this RcsbStructSymmetryRotationAxes.

        The number of times (order of rotation) that a subunit can be repeated by a rotation operation, being transformed into a new state indistinguishable from its starting state.  # noqa: E501

        :param order: The order of this RcsbStructSymmetryRotationAxes.  # noqa: E501
        :type: int
        """

        self._order = order

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
        if issubclass(RcsbStructSymmetryRotationAxes, dict):
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
        if not isinstance(other, RcsbStructSymmetryRotationAxes):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other