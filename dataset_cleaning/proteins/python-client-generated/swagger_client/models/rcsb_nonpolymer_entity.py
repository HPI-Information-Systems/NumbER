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

class RcsbNonpolymerEntity(object):
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
        'formula_weight': 'float',
        'pdbx_description': 'str',
        'pdbx_number_of_molecules': 'int'
    }

    attribute_map = {
        'details': 'details',
        'formula_weight': 'formula_weight',
        'pdbx_description': 'pdbx_description',
        'pdbx_number_of_molecules': 'pdbx_number_of_molecules'
    }

    def __init__(self, details=None, formula_weight=None, pdbx_description=None, pdbx_number_of_molecules=None):  # noqa: E501
        """RcsbNonpolymerEntity - a model defined in Swagger"""  # noqa: E501
        self._details = None
        self._formula_weight = None
        self._pdbx_description = None
        self._pdbx_number_of_molecules = None
        self.discriminator = None
        if details is not None:
            self.details = details
        if formula_weight is not None:
            self.formula_weight = formula_weight
        if pdbx_description is not None:
            self.pdbx_description = pdbx_description
        if pdbx_number_of_molecules is not None:
            self.pdbx_number_of_molecules = pdbx_number_of_molecules

    @property
    def details(self):
        """Gets the details of this RcsbNonpolymerEntity.  # noqa: E501

        A description of special aspects of the entity.  # noqa: E501

        :return: The details of this RcsbNonpolymerEntity.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this RcsbNonpolymerEntity.

        A description of special aspects of the entity.  # noqa: E501

        :param details: The details of this RcsbNonpolymerEntity.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def formula_weight(self):
        """Gets the formula_weight of this RcsbNonpolymerEntity.  # noqa: E501

        Formula mass (KDa) of the entity.  # noqa: E501

        :return: The formula_weight of this RcsbNonpolymerEntity.  # noqa: E501
        :rtype: float
        """
        return self._formula_weight

    @formula_weight.setter
    def formula_weight(self, formula_weight):
        """Sets the formula_weight of this RcsbNonpolymerEntity.

        Formula mass (KDa) of the entity.  # noqa: E501

        :param formula_weight: The formula_weight of this RcsbNonpolymerEntity.  # noqa: E501
        :type: float
        """

        self._formula_weight = formula_weight

    @property
    def pdbx_description(self):
        """Gets the pdbx_description of this RcsbNonpolymerEntity.  # noqa: E501

        A description of the nonpolymer entity.  # noqa: E501

        :return: The pdbx_description of this RcsbNonpolymerEntity.  # noqa: E501
        :rtype: str
        """
        return self._pdbx_description

    @pdbx_description.setter
    def pdbx_description(self, pdbx_description):
        """Sets the pdbx_description of this RcsbNonpolymerEntity.

        A description of the nonpolymer entity.  # noqa: E501

        :param pdbx_description: The pdbx_description of this RcsbNonpolymerEntity.  # noqa: E501
        :type: str
        """

        self._pdbx_description = pdbx_description

    @property
    def pdbx_number_of_molecules(self):
        """Gets the pdbx_number_of_molecules of this RcsbNonpolymerEntity.  # noqa: E501

        The number of molecules of the nonpolymer entity in the entry.  # noqa: E501

        :return: The pdbx_number_of_molecules of this RcsbNonpolymerEntity.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_number_of_molecules

    @pdbx_number_of_molecules.setter
    def pdbx_number_of_molecules(self, pdbx_number_of_molecules):
        """Sets the pdbx_number_of_molecules of this RcsbNonpolymerEntity.

        The number of molecules of the nonpolymer entity in the entry.  # noqa: E501

        :param pdbx_number_of_molecules: The pdbx_number_of_molecules of this RcsbNonpolymerEntity.  # noqa: E501
        :type: int
        """

        self._pdbx_number_of_molecules = pdbx_number_of_molecules

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
        if issubclass(RcsbNonpolymerEntity, dict):
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
        if not isinstance(other, RcsbNonpolymerEntity):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
