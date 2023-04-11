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

class RcsbPolymerInstanceFeatureAdditionalProperties(object):
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
        'name': 'str',
        'values': 'list[object]'
    }

    attribute_map = {
        'name': 'name',
        'values': 'values'
    }

    def __init__(self, name=None, values=None):  # noqa: E501
        """RcsbPolymerInstanceFeatureAdditionalProperties - a model defined in Swagger"""  # noqa: E501
        self._name = None
        self._values = None
        self.discriminator = None
        if name is not None:
            self.name = name
        if values is not None:
            self.values = values

    @property
    def name(self):
        """Gets the name of this RcsbPolymerInstanceFeatureAdditionalProperties.  # noqa: E501

        The additional property name.  # noqa: E501

        :return: The name of this RcsbPolymerInstanceFeatureAdditionalProperties.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this RcsbPolymerInstanceFeatureAdditionalProperties.

        The additional property name.  # noqa: E501

        :param name: The name of this RcsbPolymerInstanceFeatureAdditionalProperties.  # noqa: E501
        :type: str
        """
        allowed_values = ["CATH_DOMAIN_ID", "CATH_NAME", "ECOD_DOMAIN_ID", "ECOD_FAMILY_NAME", "MODELCIF_MODEL_ID", "OMEGA_ANGLE", "PARTNER_ASYM_ID", "PARTNER_BOND_DISTANCE", "PARTNER_COMP_ID", "SCOP2_DOMAIN_ID", "SCOP2_FAMILY_ID", "SCOP2_FAMILY_NAME", "SCOP2_SUPERFAMILY_ID", "SCOP2_SUPERFAMILY_NAME", "SCOP_DOMAIN_ID", "SCOP_NAME", "SCOP_SUN_ID", "SHEET_SENSE"]  # noqa: E501
        if name not in allowed_values:
            raise ValueError(
                "Invalid value for `name` ({0}), must be one of {1}"  # noqa: E501
                .format(name, allowed_values)
            )

        self._name = name

    @property
    def values(self):
        """Gets the values of this RcsbPolymerInstanceFeatureAdditionalProperties.  # noqa: E501

        The value(s) of the additional property.  # noqa: E501

        :return: The values of this RcsbPolymerInstanceFeatureAdditionalProperties.  # noqa: E501
        :rtype: list[object]
        """
        return self._values

    @values.setter
    def values(self, values):
        """Sets the values of this RcsbPolymerInstanceFeatureAdditionalProperties.

        The value(s) of the additional property.  # noqa: E501

        :param values: The values of this RcsbPolymerInstanceFeatureAdditionalProperties.  # noqa: E501
        :type: list[object]
        """

        self._values = values

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
        if issubclass(RcsbPolymerInstanceFeatureAdditionalProperties, dict):
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
        if not isinstance(other, RcsbPolymerInstanceFeatureAdditionalProperties):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
