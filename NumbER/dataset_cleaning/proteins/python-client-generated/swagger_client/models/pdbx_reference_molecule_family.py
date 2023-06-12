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

class PdbxReferenceMoleculeFamily(object):
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
        'family_prd_id': 'str',
        'name': 'str',
        'release_status': 'str',
        'replaced_by': 'str',
        'replaces': 'str'
    }

    attribute_map = {
        'family_prd_id': 'family_prd_id',
        'name': 'name',
        'release_status': 'release_status',
        'replaced_by': 'replaced_by',
        'replaces': 'replaces'
    }

    def __init__(self, family_prd_id=None, name=None, release_status=None, replaced_by=None, replaces=None):  # noqa: E501
        """PdbxReferenceMoleculeFamily - a model defined in Swagger"""  # noqa: E501
        self._family_prd_id = None
        self._name = None
        self._release_status = None
        self._replaced_by = None
        self._replaces = None
        self.discriminator = None
        self.family_prd_id = family_prd_id
        if name is not None:
            self.name = name
        if release_status is not None:
            self.release_status = release_status
        if replaced_by is not None:
            self.replaced_by = replaced_by
        if replaces is not None:
            self.replaces = replaces

    @property
    def family_prd_id(self):
        """Gets the family_prd_id of this PdbxReferenceMoleculeFamily.  # noqa: E501

        The value of _pdbx_reference_entity.family_prd_id must uniquely identify a record in the  PDBX_REFERENCE_MOLECULE_FAMILY list.   By convention this ID uniquely identifies the reference family in  in the PDB reference dictionary.   The ID has the template form FAM_dddddd (e.g. FAM_000001)  # noqa: E501

        :return: The family_prd_id of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :rtype: str
        """
        return self._family_prd_id

    @family_prd_id.setter
    def family_prd_id(self, family_prd_id):
        """Sets the family_prd_id of this PdbxReferenceMoleculeFamily.

        The value of _pdbx_reference_entity.family_prd_id must uniquely identify a record in the  PDBX_REFERENCE_MOLECULE_FAMILY list.   By convention this ID uniquely identifies the reference family in  in the PDB reference dictionary.   The ID has the template form FAM_dddddd (e.g. FAM_000001)  # noqa: E501

        :param family_prd_id: The family_prd_id of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :type: str
        """
        if family_prd_id is None:
            raise ValueError("Invalid value for `family_prd_id`, must not be `None`")  # noqa: E501

        self._family_prd_id = family_prd_id

    @property
    def name(self):
        """Gets the name of this PdbxReferenceMoleculeFamily.  # noqa: E501

        The entity family name.  # noqa: E501

        :return: The name of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this PdbxReferenceMoleculeFamily.

        The entity family name.  # noqa: E501

        :param name: The name of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def release_status(self):
        """Gets the release_status of this PdbxReferenceMoleculeFamily.  # noqa: E501

        Assigns the current PDB release status for this family.  # noqa: E501

        :return: The release_status of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :rtype: str
        """
        return self._release_status

    @release_status.setter
    def release_status(self, release_status):
        """Sets the release_status of this PdbxReferenceMoleculeFamily.

        Assigns the current PDB release status for this family.  # noqa: E501

        :param release_status: The release_status of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :type: str
        """
        allowed_values = ["HOLD", "OBS", "REL", "WAIT"]  # noqa: E501
        if release_status not in allowed_values:
            raise ValueError(
                "Invalid value for `release_status` ({0}), must be one of {1}"  # noqa: E501
                .format(release_status, allowed_values)
            )

        self._release_status = release_status

    @property
    def replaced_by(self):
        """Gets the replaced_by of this PdbxReferenceMoleculeFamily.  # noqa: E501

        Assigns the identifier of the family that has replaced this component.  # noqa: E501

        :return: The replaced_by of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :rtype: str
        """
        return self._replaced_by

    @replaced_by.setter
    def replaced_by(self, replaced_by):
        """Sets the replaced_by of this PdbxReferenceMoleculeFamily.

        Assigns the identifier of the family that has replaced this component.  # noqa: E501

        :param replaced_by: The replaced_by of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :type: str
        """

        self._replaced_by = replaced_by

    @property
    def replaces(self):
        """Gets the replaces of this PdbxReferenceMoleculeFamily.  # noqa: E501

        Assigns the identifier for the family which have been replaced by this family.  Multiple family identifier codes should be separated by commas.  # noqa: E501

        :return: The replaces of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :rtype: str
        """
        return self._replaces

    @replaces.setter
    def replaces(self, replaces):
        """Sets the replaces of this PdbxReferenceMoleculeFamily.

        Assigns the identifier for the family which have been replaced by this family.  Multiple family identifier codes should be separated by commas.  # noqa: E501

        :param replaces: The replaces of this PdbxReferenceMoleculeFamily.  # noqa: E501
        :type: str
        """

        self._replaces = replaces

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
        if issubclass(PdbxReferenceMoleculeFamily, dict):
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
        if not isinstance(other, PdbxReferenceMoleculeFamily):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other