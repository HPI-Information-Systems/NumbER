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

class RcsbChemCompTarget(object):
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
        'comp_id': 'str',
        'interaction_type': 'str',
        'name': 'str',
        'ordinal': 'int',
        'provenance_source': 'str',
        'reference_database_accession_code': 'str',
        'reference_database_name': 'str',
        'target_actions': 'list[str]'
    }

    attribute_map = {
        'comp_id': 'comp_id',
        'interaction_type': 'interaction_type',
        'name': 'name',
        'ordinal': 'ordinal',
        'provenance_source': 'provenance_source',
        'reference_database_accession_code': 'reference_database_accession_code',
        'reference_database_name': 'reference_database_name',
        'target_actions': 'target_actions'
    }

    def __init__(self, comp_id=None, interaction_type=None, name=None, ordinal=None, provenance_source=None, reference_database_accession_code=None, reference_database_name=None, target_actions=None):  # noqa: E501
        """RcsbChemCompTarget - a model defined in Swagger"""  # noqa: E501
        self._comp_id = None
        self._interaction_type = None
        self._name = None
        self._ordinal = None
        self._provenance_source = None
        self._reference_database_accession_code = None
        self._reference_database_name = None
        self._target_actions = None
        self.discriminator = None
        self.comp_id = comp_id
        if interaction_type is not None:
            self.interaction_type = interaction_type
        if name is not None:
            self.name = name
        self.ordinal = ordinal
        if provenance_source is not None:
            self.provenance_source = provenance_source
        if reference_database_accession_code is not None:
            self.reference_database_accession_code = reference_database_accession_code
        if reference_database_name is not None:
            self.reference_database_name = reference_database_name
        if target_actions is not None:
            self.target_actions = target_actions

    @property
    def comp_id(self):
        """Gets the comp_id of this RcsbChemCompTarget.  # noqa: E501

        The value of _rcsb_chem_comp_target.comp_id is a reference to  a chemical component definition.  # noqa: E501

        :return: The comp_id of this RcsbChemCompTarget.  # noqa: E501
        :rtype: str
        """
        return self._comp_id

    @comp_id.setter
    def comp_id(self, comp_id):
        """Sets the comp_id of this RcsbChemCompTarget.

        The value of _rcsb_chem_comp_target.comp_id is a reference to  a chemical component definition.  # noqa: E501

        :param comp_id: The comp_id of this RcsbChemCompTarget.  # noqa: E501
        :type: str
        """
        if comp_id is None:
            raise ValueError("Invalid value for `comp_id`, must not be `None`")  # noqa: E501

        self._comp_id = comp_id

    @property
    def interaction_type(self):
        """Gets the interaction_type of this RcsbChemCompTarget.  # noqa: E501

        The type of target interaction.  # noqa: E501

        :return: The interaction_type of this RcsbChemCompTarget.  # noqa: E501
        :rtype: str
        """
        return self._interaction_type

    @interaction_type.setter
    def interaction_type(self, interaction_type):
        """Sets the interaction_type of this RcsbChemCompTarget.

        The type of target interaction.  # noqa: E501

        :param interaction_type: The interaction_type of this RcsbChemCompTarget.  # noqa: E501
        :type: str
        """

        self._interaction_type = interaction_type

    @property
    def name(self):
        """Gets the name of this RcsbChemCompTarget.  # noqa: E501

        The chemical component target name.  # noqa: E501

        :return: The name of this RcsbChemCompTarget.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this RcsbChemCompTarget.

        The chemical component target name.  # noqa: E501

        :param name: The name of this RcsbChemCompTarget.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def ordinal(self):
        """Gets the ordinal of this RcsbChemCompTarget.  # noqa: E501

        The value of _rcsb_chem_comp_target.ordinal distinguishes  related examples for each chemical component.  # noqa: E501

        :return: The ordinal of this RcsbChemCompTarget.  # noqa: E501
        :rtype: int
        """
        return self._ordinal

    @ordinal.setter
    def ordinal(self, ordinal):
        """Sets the ordinal of this RcsbChemCompTarget.

        The value of _rcsb_chem_comp_target.ordinal distinguishes  related examples for each chemical component.  # noqa: E501

        :param ordinal: The ordinal of this RcsbChemCompTarget.  # noqa: E501
        :type: int
        """
        if ordinal is None:
            raise ValueError("Invalid value for `ordinal`, must not be `None`")  # noqa: E501

        self._ordinal = ordinal

    @property
    def provenance_source(self):
        """Gets the provenance_source of this RcsbChemCompTarget.  # noqa: E501

        A code indicating the provenance of the target interaction assignment  # noqa: E501

        :return: The provenance_source of this RcsbChemCompTarget.  # noqa: E501
        :rtype: str
        """
        return self._provenance_source

    @provenance_source.setter
    def provenance_source(self, provenance_source):
        """Sets the provenance_source of this RcsbChemCompTarget.

        A code indicating the provenance of the target interaction assignment  # noqa: E501

        :param provenance_source: The provenance_source of this RcsbChemCompTarget.  # noqa: E501
        :type: str
        """
        allowed_values = ["DrugBank", "PDB Primary Data"]  # noqa: E501
        if provenance_source not in allowed_values:
            raise ValueError(
                "Invalid value for `provenance_source` ({0}), must be one of {1}"  # noqa: E501
                .format(provenance_source, allowed_values)
            )

        self._provenance_source = provenance_source

    @property
    def reference_database_accession_code(self):
        """Gets the reference_database_accession_code of this RcsbChemCompTarget.  # noqa: E501

        The reference identifier code for the target interaction reference.  # noqa: E501

        :return: The reference_database_accession_code of this RcsbChemCompTarget.  # noqa: E501
        :rtype: str
        """
        return self._reference_database_accession_code

    @reference_database_accession_code.setter
    def reference_database_accession_code(self, reference_database_accession_code):
        """Sets the reference_database_accession_code of this RcsbChemCompTarget.

        The reference identifier code for the target interaction reference.  # noqa: E501

        :param reference_database_accession_code: The reference_database_accession_code of this RcsbChemCompTarget.  # noqa: E501
        :type: str
        """

        self._reference_database_accession_code = reference_database_accession_code

    @property
    def reference_database_name(self):
        """Gets the reference_database_name of this RcsbChemCompTarget.  # noqa: E501

        The reference database name for the target interaction.  # noqa: E501

        :return: The reference_database_name of this RcsbChemCompTarget.  # noqa: E501
        :rtype: str
        """
        return self._reference_database_name

    @reference_database_name.setter
    def reference_database_name(self, reference_database_name):
        """Sets the reference_database_name of this RcsbChemCompTarget.

        The reference database name for the target interaction.  # noqa: E501

        :param reference_database_name: The reference_database_name of this RcsbChemCompTarget.  # noqa: E501
        :type: str
        """
        allowed_values = ["UniProt"]  # noqa: E501
        if reference_database_name not in allowed_values:
            raise ValueError(
                "Invalid value for `reference_database_name` ({0}), must be one of {1}"  # noqa: E501
                .format(reference_database_name, allowed_values)
            )

        self._reference_database_name = reference_database_name

    @property
    def target_actions(self):
        """Gets the target_actions of this RcsbChemCompTarget.  # noqa: E501

        The mechanism of action of the chemical component - target interaction.  # noqa: E501

        :return: The target_actions of this RcsbChemCompTarget.  # noqa: E501
        :rtype: list[str]
        """
        return self._target_actions

    @target_actions.setter
    def target_actions(self, target_actions):
        """Sets the target_actions of this RcsbChemCompTarget.

        The mechanism of action of the chemical component - target interaction.  # noqa: E501

        :param target_actions: The target_actions of this RcsbChemCompTarget.  # noqa: E501
        :type: list[str]
        """

        self._target_actions = target_actions

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
        if issubclass(RcsbChemCompTarget, dict):
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
        if not isinstance(other, RcsbChemCompTarget):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other