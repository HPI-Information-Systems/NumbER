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

class PdbxSerialCrystallographySampleDelivery(object):
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
        'description': 'str',
        'diffrn_id': 'str',
        'method': 'str'
    }

    attribute_map = {
        'description': 'description',
        'diffrn_id': 'diffrn_id',
        'method': 'method'
    }

    def __init__(self, description=None, diffrn_id=None, method=None):  # noqa: E501
        """PdbxSerialCrystallographySampleDelivery - a model defined in Swagger"""  # noqa: E501
        self._description = None
        self._diffrn_id = None
        self._method = None
        self.discriminator = None
        if description is not None:
            self.description = description
        self.diffrn_id = diffrn_id
        if method is not None:
            self.method = method

    @property
    def description(self):
        """Gets the description of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501

        The description of the mechanism by which the specimen in placed in the path  of the source.  # noqa: E501

        :return: The description of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this PdbxSerialCrystallographySampleDelivery.

        The description of the mechanism by which the specimen in placed in the path  of the source.  # noqa: E501

        :param description: The description of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def diffrn_id(self):
        """Gets the diffrn_id of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501

        The data item is a pointer to _diffrn.id in the DIFFRN  category.  # noqa: E501

        :return: The diffrn_id of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501
        :rtype: str
        """
        return self._diffrn_id

    @diffrn_id.setter
    def diffrn_id(self, diffrn_id):
        """Sets the diffrn_id of this PdbxSerialCrystallographySampleDelivery.

        The data item is a pointer to _diffrn.id in the DIFFRN  category.  # noqa: E501

        :param diffrn_id: The diffrn_id of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501
        :type: str
        """
        if diffrn_id is None:
            raise ValueError("Invalid value for `diffrn_id`, must not be `None`")  # noqa: E501

        self._diffrn_id = diffrn_id

    @property
    def method(self):
        """Gets the method of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501

        The description of the mechanism by which the specimen in placed in the path  of the source.  # noqa: E501

        :return: The method of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501
        :rtype: str
        """
        return self._method

    @method.setter
    def method(self, method):
        """Sets the method of this PdbxSerialCrystallographySampleDelivery.

        The description of the mechanism by which the specimen in placed in the path  of the source.  # noqa: E501

        :param method: The method of this PdbxSerialCrystallographySampleDelivery.  # noqa: E501
        :type: str
        """
        allowed_values = ["fixed target", "injection"]  # noqa: E501
        if method not in allowed_values:
            raise ValueError(
                "Invalid value for `method` ({0}), must be one of {1}"  # noqa: E501
                .format(method, allowed_values)
            )

        self._method = method

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
        if issubclass(PdbxSerialCrystallographySampleDelivery, dict):
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
        if not isinstance(other, PdbxSerialCrystallographySampleDelivery):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other