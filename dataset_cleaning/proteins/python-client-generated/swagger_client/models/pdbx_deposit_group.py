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

class PdbxDepositGroup(object):
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
        'group_description': 'str',
        'group_id': 'str',
        'group_title': 'str',
        'group_type': 'str'
    }

    attribute_map = {
        'group_description': 'group_description',
        'group_id': 'group_id',
        'group_title': 'group_title',
        'group_type': 'group_type'
    }

    def __init__(self, group_description=None, group_id=None, group_title=None, group_type=None):  # noqa: E501
        """PdbxDepositGroup - a model defined in Swagger"""  # noqa: E501
        self._group_description = None
        self._group_id = None
        self._group_title = None
        self._group_type = None
        self.discriminator = None
        if group_description is not None:
            self.group_description = group_description
        self.group_id = group_id
        if group_title is not None:
            self.group_title = group_title
        if group_type is not None:
            self.group_type = group_type

    @property
    def group_description(self):
        """Gets the group_description of this PdbxDepositGroup.  # noqa: E501

        A description of the contents of entries in the collection.  # noqa: E501

        :return: The group_description of this PdbxDepositGroup.  # noqa: E501
        :rtype: str
        """
        return self._group_description

    @group_description.setter
    def group_description(self, group_description):
        """Sets the group_description of this PdbxDepositGroup.

        A description of the contents of entries in the collection.  # noqa: E501

        :param group_description: The group_description of this PdbxDepositGroup.  # noqa: E501
        :type: str
        """

        self._group_description = group_description

    @property
    def group_id(self):
        """Gets the group_id of this PdbxDepositGroup.  # noqa: E501

        A unique identifier for a group of entries deposited as a collection.  # noqa: E501

        :return: The group_id of this PdbxDepositGroup.  # noqa: E501
        :rtype: str
        """
        return self._group_id

    @group_id.setter
    def group_id(self, group_id):
        """Sets the group_id of this PdbxDepositGroup.

        A unique identifier for a group of entries deposited as a collection.  # noqa: E501

        :param group_id: The group_id of this PdbxDepositGroup.  # noqa: E501
        :type: str
        """
        if group_id is None:
            raise ValueError("Invalid value for `group_id`, must not be `None`")  # noqa: E501

        self._group_id = group_id

    @property
    def group_title(self):
        """Gets the group_title of this PdbxDepositGroup.  # noqa: E501

        A title to describe the group of entries deposited in the collection.  # noqa: E501

        :return: The group_title of this PdbxDepositGroup.  # noqa: E501
        :rtype: str
        """
        return self._group_title

    @group_title.setter
    def group_title(self, group_title):
        """Sets the group_title of this PdbxDepositGroup.

        A title to describe the group of entries deposited in the collection.  # noqa: E501

        :param group_title: The group_title of this PdbxDepositGroup.  # noqa: E501
        :type: str
        """

        self._group_title = group_title

    @property
    def group_type(self):
        """Gets the group_type of this PdbxDepositGroup.  # noqa: E501

        Text to describe a grouping of entries in multiple collections  # noqa: E501

        :return: The group_type of this PdbxDepositGroup.  # noqa: E501
        :rtype: str
        """
        return self._group_type

    @group_type.setter
    def group_type(self, group_type):
        """Sets the group_type of this PdbxDepositGroup.

        Text to describe a grouping of entries in multiple collections  # noqa: E501

        :param group_type: The group_type of this PdbxDepositGroup.  # noqa: E501
        :type: str
        """
        allowed_values = ["changed state", "ground state", "undefined"]  # noqa: E501
        if group_type not in allowed_values:
            raise ValueError(
                "Invalid value for `group_type` ({0}), must be one of {1}"  # noqa: E501
                .format(group_type, allowed_values)
            )

        self._group_type = group_type

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
        if issubclass(PdbxDepositGroup, dict):
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
        if not isinstance(other, PdbxDepositGroup):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
