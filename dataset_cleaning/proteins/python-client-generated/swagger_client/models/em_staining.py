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

class EmStaining(object):
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
        'details': 'str',
        'id': 'str',
        'material': 'str',
        'specimen_id': 'str',
        'type': 'str'
    }

    attribute_map = {
        'details': 'details',
        'id': 'id',
        'material': 'material',
        'specimen_id': 'specimen_id',
        'type': 'type'
    }

    def __init__(self, details=None, id=None, material=None, specimen_id=None, type=None):  # noqa: E501
        """EmStaining - a model defined in Swagger"""  # noqa: E501
        self._details = None
        self._id = None
        self._material = None
        self._specimen_id = None
        self._type = None
        self.discriminator = None
        if details is not None:
            self.details = details
        self.id = id
        if material is not None:
            self.material = material
        if specimen_id is not None:
            self.specimen_id = specimen_id
        if type is not None:
            self.type = type

    @property
    def details(self):
        """Gets the details of this EmStaining.  # noqa: E501

        Staining procedure used in the specimen preparation.  # noqa: E501

        :return: The details of this EmStaining.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this EmStaining.

        Staining procedure used in the specimen preparation.  # noqa: E501

        :param details: The details of this EmStaining.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def id(self):
        """Gets the id of this EmStaining.  # noqa: E501

        PRIMARY KEY  # noqa: E501

        :return: The id of this EmStaining.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this EmStaining.

        PRIMARY KEY  # noqa: E501

        :param id: The id of this EmStaining.  # noqa: E501
        :type: str
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def material(self):
        """Gets the material of this EmStaining.  # noqa: E501

        The staining  material.  # noqa: E501

        :return: The material of this EmStaining.  # noqa: E501
        :rtype: str
        """
        return self._material

    @material.setter
    def material(self, material):
        """Sets the material of this EmStaining.

        The staining  material.  # noqa: E501

        :param material: The material of this EmStaining.  # noqa: E501
        :type: str
        """

        self._material = material

    @property
    def specimen_id(self):
        """Gets the specimen_id of this EmStaining.  # noqa: E501

        Foreign key relationship to the EM SPECIMEN category  # noqa: E501

        :return: The specimen_id of this EmStaining.  # noqa: E501
        :rtype: str
        """
        return self._specimen_id

    @specimen_id.setter
    def specimen_id(self, specimen_id):
        """Sets the specimen_id of this EmStaining.

        Foreign key relationship to the EM SPECIMEN category  # noqa: E501

        :param specimen_id: The specimen_id of this EmStaining.  # noqa: E501
        :type: str
        """

        self._specimen_id = specimen_id

    @property
    def type(self):
        """Gets the type of this EmStaining.  # noqa: E501

        type of staining  # noqa: E501

        :return: The type of this EmStaining.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this EmStaining.

        type of staining  # noqa: E501

        :param type: The type of this EmStaining.  # noqa: E501
        :type: str
        """
        allowed_values = ["NEGATIVE", "NONE", "POSITIVE"]  # noqa: E501
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` ({0}), must be one of {1}"  # noqa: E501
                .format(type, allowed_values)
            )

        self._type = type

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
        if issubclass(EmStaining, dict):
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
        if not isinstance(other, EmStaining):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
