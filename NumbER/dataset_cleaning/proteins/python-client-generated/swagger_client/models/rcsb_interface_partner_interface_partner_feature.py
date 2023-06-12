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

class RcsbInterfacePartnerInterfacePartnerFeature(object):
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
        'assignment_version': 'str',
        'description': 'str',
        'feature_id': 'str',
        'name': 'str',
        'provenance_source': 'str',
        'type': 'str',
        'feature_positions': 'list[InterfacePartnerFeatureFeaturePositions]',
        'additional_properties': 'list[InterfacePartnerFeatureAdditionalProperties]'
    }

    attribute_map = {
        'assignment_version': 'assignment_version',
        'description': 'description',
        'feature_id': 'feature_id',
        'name': 'name',
        'provenance_source': 'provenance_source',
        'type': 'type',
        'feature_positions': 'feature_positions',
        'additional_properties': 'additional_properties'
    }

    def __init__(self, assignment_version=None, description=None, feature_id=None, name=None, provenance_source=None, type=None, feature_positions=None, additional_properties=None):  # noqa: E501
        """RcsbInterfacePartnerInterfacePartnerFeature - a model defined in Swagger"""  # noqa: E501
        self._assignment_version = None
        self._description = None
        self._feature_id = None
        self._name = None
        self._provenance_source = None
        self._type = None
        self._feature_positions = None
        self._additional_properties = None
        self.discriminator = None
        if assignment_version is not None:
            self.assignment_version = assignment_version
        if description is not None:
            self.description = description
        if feature_id is not None:
            self.feature_id = feature_id
        if name is not None:
            self.name = name
        if provenance_source is not None:
            self.provenance_source = provenance_source
        if type is not None:
            self.type = type
        if feature_positions is not None:
            self.feature_positions = feature_positions
        if additional_properties is not None:
            self.additional_properties = additional_properties

    @property
    def assignment_version(self):
        """Gets the assignment_version of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501

        Identifies the version of the feature assignment.  # noqa: E501

        :return: The assignment_version of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: str
        """
        return self._assignment_version

    @assignment_version.setter
    def assignment_version(self, assignment_version):
        """Sets the assignment_version of this RcsbInterfacePartnerInterfacePartnerFeature.

        Identifies the version of the feature assignment.  # noqa: E501

        :param assignment_version: The assignment_version of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: str
        """

        self._assignment_version = assignment_version

    @property
    def description(self):
        """Gets the description of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501

        A description for the feature.  # noqa: E501

        :return: The description of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this RcsbInterfacePartnerInterfacePartnerFeature.

        A description for the feature.  # noqa: E501

        :param description: The description of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def feature_id(self):
        """Gets the feature_id of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501

        An identifier for the feature.  # noqa: E501

        :return: The feature_id of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: str
        """
        return self._feature_id

    @feature_id.setter
    def feature_id(self, feature_id):
        """Sets the feature_id of this RcsbInterfacePartnerInterfacePartnerFeature.

        An identifier for the feature.  # noqa: E501

        :param feature_id: The feature_id of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: str
        """

        self._feature_id = feature_id

    @property
    def name(self):
        """Gets the name of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501

        A name for the feature.  # noqa: E501

        :return: The name of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this RcsbInterfacePartnerInterfacePartnerFeature.

        A name for the feature.  # noqa: E501

        :param name: The name of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def provenance_source(self):
        """Gets the provenance_source of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501

        Code identifying the individual, organization or program that assigned the feature.  # noqa: E501

        :return: The provenance_source of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: str
        """
        return self._provenance_source

    @provenance_source.setter
    def provenance_source(self, provenance_source):
        """Sets the provenance_source of this RcsbInterfacePartnerInterfacePartnerFeature.

        Code identifying the individual, organization or program that assigned the feature.  # noqa: E501

        :param provenance_source: The provenance_source of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: str
        """

        self._provenance_source = provenance_source

    @property
    def type(self):
        """Gets the type of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501

        A type or category of the feature.  # noqa: E501

        :return: The type of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this RcsbInterfacePartnerInterfacePartnerFeature.

        A type or category of the feature.  # noqa: E501

        :param type: The type of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: str
        """
        allowed_values = ["ASA_UNBOUND", "ASA_BOUND"]  # noqa: E501
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` ({0}), must be one of {1}"  # noqa: E501
                .format(type, allowed_values)
            )

        self._type = type

    @property
    def feature_positions(self):
        """Gets the feature_positions of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501


        :return: The feature_positions of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: list[InterfacePartnerFeatureFeaturePositions]
        """
        return self._feature_positions

    @feature_positions.setter
    def feature_positions(self, feature_positions):
        """Sets the feature_positions of this RcsbInterfacePartnerInterfacePartnerFeature.


        :param feature_positions: The feature_positions of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: list[InterfacePartnerFeatureFeaturePositions]
        """

        self._feature_positions = feature_positions

    @property
    def additional_properties(self):
        """Gets the additional_properties of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501


        :return: The additional_properties of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :rtype: list[InterfacePartnerFeatureAdditionalProperties]
        """
        return self._additional_properties

    @additional_properties.setter
    def additional_properties(self, additional_properties):
        """Sets the additional_properties of this RcsbInterfacePartnerInterfacePartnerFeature.


        :param additional_properties: The additional_properties of this RcsbInterfacePartnerInterfacePartnerFeature.  # noqa: E501
        :type: list[InterfacePartnerFeatureAdditionalProperties]
        """

        self._additional_properties = additional_properties

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
        if issubclass(RcsbInterfacePartnerInterfacePartnerFeature, dict):
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
        if not isinstance(other, RcsbInterfacePartnerInterfacePartnerFeature):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other