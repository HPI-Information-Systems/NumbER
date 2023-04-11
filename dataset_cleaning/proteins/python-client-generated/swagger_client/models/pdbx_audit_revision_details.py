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

class PdbxAuditRevisionDetails(object):
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
        'data_content_type': 'str',
        'description': 'str',
        'details': 'str',
        'ordinal': 'int',
        'provider': 'str',
        'revision_ordinal': 'int',
        'type': 'str'
    }

    attribute_map = {
        'data_content_type': 'data_content_type',
        'description': 'description',
        'details': 'details',
        'ordinal': 'ordinal',
        'provider': 'provider',
        'revision_ordinal': 'revision_ordinal',
        'type': 'type'
    }

    def __init__(self, data_content_type=None, description=None, details=None, ordinal=None, provider=None, revision_ordinal=None, type=None):  # noqa: E501
        """PdbxAuditRevisionDetails - a model defined in Swagger"""  # noqa: E501
        self._data_content_type = None
        self._description = None
        self._details = None
        self._ordinal = None
        self._provider = None
        self._revision_ordinal = None
        self._type = None
        self.discriminator = None
        self.data_content_type = data_content_type
        if description is not None:
            self.description = description
        if details is not None:
            self.details = details
        self.ordinal = ordinal
        if provider is not None:
            self.provider = provider
        self.revision_ordinal = revision_ordinal
        if type is not None:
            self.type = type

    @property
    def data_content_type(self):
        """Gets the data_content_type of this PdbxAuditRevisionDetails.  # noqa: E501

        The type of file that the pdbx_audit_revision_history record refers to.  # noqa: E501

        :return: The data_content_type of this PdbxAuditRevisionDetails.  # noqa: E501
        :rtype: str
        """
        return self._data_content_type

    @data_content_type.setter
    def data_content_type(self, data_content_type):
        """Sets the data_content_type of this PdbxAuditRevisionDetails.

        The type of file that the pdbx_audit_revision_history record refers to.  # noqa: E501

        :param data_content_type: The data_content_type of this PdbxAuditRevisionDetails.  # noqa: E501
        :type: str
        """
        if data_content_type is None:
            raise ValueError("Invalid value for `data_content_type`, must not be `None`")  # noqa: E501
        allowed_values = ["Chemical component", "NMR restraints", "NMR shifts", "Structure factors", "Structure model"]  # noqa: E501
        if data_content_type not in allowed_values:
            raise ValueError(
                "Invalid value for `data_content_type` ({0}), must be one of {1}"  # noqa: E501
                .format(data_content_type, allowed_values)
            )

        self._data_content_type = data_content_type

    @property
    def description(self):
        """Gets the description of this PdbxAuditRevisionDetails.  # noqa: E501

        Additional details describing the revision.  # noqa: E501

        :return: The description of this PdbxAuditRevisionDetails.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this PdbxAuditRevisionDetails.

        Additional details describing the revision.  # noqa: E501

        :param description: The description of this PdbxAuditRevisionDetails.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def details(self):
        """Gets the details of this PdbxAuditRevisionDetails.  # noqa: E501

        Further details describing the revision.  # noqa: E501

        :return: The details of this PdbxAuditRevisionDetails.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this PdbxAuditRevisionDetails.

        Further details describing the revision.  # noqa: E501

        :param details: The details of this PdbxAuditRevisionDetails.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def ordinal(self):
        """Gets the ordinal of this PdbxAuditRevisionDetails.  # noqa: E501

        A unique identifier for the pdbx_audit_revision_details record.  # noqa: E501

        :return: The ordinal of this PdbxAuditRevisionDetails.  # noqa: E501
        :rtype: int
        """
        return self._ordinal

    @ordinal.setter
    def ordinal(self, ordinal):
        """Sets the ordinal of this PdbxAuditRevisionDetails.

        A unique identifier for the pdbx_audit_revision_details record.  # noqa: E501

        :param ordinal: The ordinal of this PdbxAuditRevisionDetails.  # noqa: E501
        :type: int
        """
        if ordinal is None:
            raise ValueError("Invalid value for `ordinal`, must not be `None`")  # noqa: E501

        self._ordinal = ordinal

    @property
    def provider(self):
        """Gets the provider of this PdbxAuditRevisionDetails.  # noqa: E501

        The provider of the revision.  # noqa: E501

        :return: The provider of this PdbxAuditRevisionDetails.  # noqa: E501
        :rtype: str
        """
        return self._provider

    @provider.setter
    def provider(self, provider):
        """Sets the provider of this PdbxAuditRevisionDetails.

        The provider of the revision.  # noqa: E501

        :param provider: The provider of this PdbxAuditRevisionDetails.  # noqa: E501
        :type: str
        """
        allowed_values = ["author", "repository"]  # noqa: E501
        if provider not in allowed_values:
            raise ValueError(
                "Invalid value for `provider` ({0}), must be one of {1}"  # noqa: E501
                .format(provider, allowed_values)
            )

        self._provider = provider

    @property
    def revision_ordinal(self):
        """Gets the revision_ordinal of this PdbxAuditRevisionDetails.  # noqa: E501

        A pointer to  _pdbx_audit_revision_history.ordinal  # noqa: E501

        :return: The revision_ordinal of this PdbxAuditRevisionDetails.  # noqa: E501
        :rtype: int
        """
        return self._revision_ordinal

    @revision_ordinal.setter
    def revision_ordinal(self, revision_ordinal):
        """Sets the revision_ordinal of this PdbxAuditRevisionDetails.

        A pointer to  _pdbx_audit_revision_history.ordinal  # noqa: E501

        :param revision_ordinal: The revision_ordinal of this PdbxAuditRevisionDetails.  # noqa: E501
        :type: int
        """
        if revision_ordinal is None:
            raise ValueError("Invalid value for `revision_ordinal`, must not be `None`")  # noqa: E501

        self._revision_ordinal = revision_ordinal

    @property
    def type(self):
        """Gets the type of this PdbxAuditRevisionDetails.  # noqa: E501

        A type classification of the revision  # noqa: E501

        :return: The type of this PdbxAuditRevisionDetails.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this PdbxAuditRevisionDetails.

        A type classification of the revision  # noqa: E501

        :param type: The type of this PdbxAuditRevisionDetails.  # noqa: E501
        :type: str
        """
        allowed_values = ["Coordinate replacement", "Initial release", "Obsolete", "Remediation"]  # noqa: E501
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
        if issubclass(PdbxAuditRevisionDetails, dict):
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
        if not isinstance(other, PdbxAuditRevisionDetails):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other