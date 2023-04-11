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

class RcsbMembraneLineage(object):
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
        'id': 'str',
        'name': 'str',
        'depth': 'int'
    }

    attribute_map = {
        'id': 'id',
        'name': 'name',
        'depth': 'depth'
    }

    def __init__(self, id=None, name=None, depth=None):  # noqa: E501
        """RcsbMembraneLineage - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._name = None
        self._depth = None
        self.discriminator = None
        if id is not None:
            self.id = id
        if name is not None:
            self.name = name
        if depth is not None:
            self.depth = depth

    @property
    def id(self):
        """Gets the id of this RcsbMembraneLineage.  # noqa: E501

        Automatically assigned ID for membrane classification term in the Membrane Protein Browser.  # noqa: E501

        :return: The id of this RcsbMembraneLineage.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this RcsbMembraneLineage.

        Automatically assigned ID for membrane classification term in the Membrane Protein Browser.  # noqa: E501

        :param id: The id of this RcsbMembraneLineage.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def name(self):
        """Gets the name of this RcsbMembraneLineage.  # noqa: E501

        Membrane protein classification term.  # noqa: E501

        :return: The name of this RcsbMembraneLineage.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this RcsbMembraneLineage.

        Membrane protein classification term.  # noqa: E501

        :param name: The name of this RcsbMembraneLineage.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def depth(self):
        """Gets the depth of this RcsbMembraneLineage.  # noqa: E501

        Hierarchy depth.  # noqa: E501

        :return: The depth of this RcsbMembraneLineage.  # noqa: E501
        :rtype: int
        """
        return self._depth

    @depth.setter
    def depth(self, depth):
        """Sets the depth of this RcsbMembraneLineage.

        Hierarchy depth.  # noqa: E501

        :param depth: The depth of this RcsbMembraneLineage.  # noqa: E501
        :type: int
        """

        self._depth = depth

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
        if issubclass(RcsbMembraneLineage, dict):
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
        if not isinstance(other, RcsbMembraneLineage):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other