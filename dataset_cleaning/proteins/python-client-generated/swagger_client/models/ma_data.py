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

class MaData(object):
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
        'content_type': 'str',
        'content_type_other_details': 'str',
        'id': 'int',
        'name': 'str'
    }

    attribute_map = {
        'content_type': 'content_type',
        'content_type_other_details': 'content_type_other_details',
        'id': 'id',
        'name': 'name'
    }

    def __init__(self, content_type=None, content_type_other_details=None, id=None, name=None):  # noqa: E501
        """MaData - a model defined in Swagger"""  # noqa: E501
        self._content_type = None
        self._content_type_other_details = None
        self._id = None
        self._name = None
        self.discriminator = None
        if content_type is not None:
            self.content_type = content_type
        if content_type_other_details is not None:
            self.content_type_other_details = content_type_other_details
        self.id = id
        if name is not None:
            self.name = name

    @property
    def content_type(self):
        """Gets the content_type of this MaData.  # noqa: E501

        The type of data held in the dataset.  # noqa: E501

        :return: The content_type of this MaData.  # noqa: E501
        :rtype: str
        """
        return self._content_type

    @content_type.setter
    def content_type(self, content_type):
        """Sets the content_type of this MaData.

        The type of data held in the dataset.  # noqa: E501

        :param content_type: The content_type of this MaData.  # noqa: E501
        :type: str
        """
        allowed_values = ["coevolution MSA", "input structure", "model coordinates", "other", "polymeric template library", "reference database", "spatial restraints", "target", "target-template alignment", "template structure"]  # noqa: E501
        if content_type not in allowed_values:
            raise ValueError(
                "Invalid value for `content_type` ({0}), must be one of {1}"  # noqa: E501
                .format(content_type, allowed_values)
            )

        self._content_type = content_type

    @property
    def content_type_other_details(self):
        """Gets the content_type_other_details of this MaData.  # noqa: E501

        Details for other content types.  # noqa: E501

        :return: The content_type_other_details of this MaData.  # noqa: E501
        :rtype: str
        """
        return self._content_type_other_details

    @content_type_other_details.setter
    def content_type_other_details(self, content_type_other_details):
        """Sets the content_type_other_details of this MaData.

        Details for other content types.  # noqa: E501

        :param content_type_other_details: The content_type_other_details of this MaData.  # noqa: E501
        :type: str
        """

        self._content_type_other_details = content_type_other_details

    @property
    def id(self):
        """Gets the id of this MaData.  # noqa: E501

        A unique identifier for the data.  # noqa: E501

        :return: The id of this MaData.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this MaData.

        A unique identifier for the data.  # noqa: E501

        :param id: The id of this MaData.  # noqa: E501
        :type: int
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def name(self):
        """Gets the name of this MaData.  # noqa: E501

        An author-given name for the content held in the dataset.  # noqa: E501

        :return: The name of this MaData.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this MaData.

        An author-given name for the content held in the dataset.  # noqa: E501

        :param name: The name of this MaData.  # noqa: E501
        :type: str
        """

        self._name = name

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
        if issubclass(MaData, dict):
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
        if not isinstance(other, MaData):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
