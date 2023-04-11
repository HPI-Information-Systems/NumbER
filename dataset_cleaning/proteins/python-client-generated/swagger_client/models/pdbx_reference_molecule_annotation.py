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

class PdbxReferenceMoleculeAnnotation(object):
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
        'ordinal': 'int',
        'prd_id': 'str',
        'source': 'str',
        'text': 'str',
        'type': 'str'
    }

    attribute_map = {
        'family_prd_id': 'family_prd_id',
        'ordinal': 'ordinal',
        'prd_id': 'prd_id',
        'source': 'source',
        'text': 'text',
        'type': 'type'
    }

    def __init__(self, family_prd_id=None, ordinal=None, prd_id=None, source=None, text=None, type=None):  # noqa: E501
        """PdbxReferenceMoleculeAnnotation - a model defined in Swagger"""  # noqa: E501
        self._family_prd_id = None
        self._ordinal = None
        self._prd_id = None
        self._source = None
        self._text = None
        self._type = None
        self.discriminator = None
        self.family_prd_id = family_prd_id
        self.ordinal = ordinal
        if prd_id is not None:
            self.prd_id = prd_id
        if source is not None:
            self.source = source
        if text is not None:
            self.text = text
        if type is not None:
            self.type = type

    @property
    def family_prd_id(self):
        """Gets the family_prd_id of this PdbxReferenceMoleculeAnnotation.  # noqa: E501

        The value of _pdbx_reference_molecule_annotation.family_prd_id is a reference to  _pdbx_reference_molecule_list.family_prd_id in category PDBX_REFERENCE_MOLECULE_FAMILY_LIST.  # noqa: E501

        :return: The family_prd_id of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :rtype: str
        """
        return self._family_prd_id

    @family_prd_id.setter
    def family_prd_id(self, family_prd_id):
        """Sets the family_prd_id of this PdbxReferenceMoleculeAnnotation.

        The value of _pdbx_reference_molecule_annotation.family_prd_id is a reference to  _pdbx_reference_molecule_list.family_prd_id in category PDBX_REFERENCE_MOLECULE_FAMILY_LIST.  # noqa: E501

        :param family_prd_id: The family_prd_id of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :type: str
        """
        if family_prd_id is None:
            raise ValueError("Invalid value for `family_prd_id`, must not be `None`")  # noqa: E501

        self._family_prd_id = family_prd_id

    @property
    def ordinal(self):
        """Gets the ordinal of this PdbxReferenceMoleculeAnnotation.  # noqa: E501

        This data item distinguishes anotations for this entity.  # noqa: E501

        :return: The ordinal of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :rtype: int
        """
        return self._ordinal

    @ordinal.setter
    def ordinal(self, ordinal):
        """Sets the ordinal of this PdbxReferenceMoleculeAnnotation.

        This data item distinguishes anotations for this entity.  # noqa: E501

        :param ordinal: The ordinal of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :type: int
        """
        if ordinal is None:
            raise ValueError("Invalid value for `ordinal`, must not be `None`")  # noqa: E501

        self._ordinal = ordinal

    @property
    def prd_id(self):
        """Gets the prd_id of this PdbxReferenceMoleculeAnnotation.  # noqa: E501

        This data item is a pointer to _pdbx_reference_molecule.prd_id in the  PDB_REFERENCE_MOLECULE category.  # noqa: E501

        :return: The prd_id of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :rtype: str
        """
        return self._prd_id

    @prd_id.setter
    def prd_id(self, prd_id):
        """Sets the prd_id of this PdbxReferenceMoleculeAnnotation.

        This data item is a pointer to _pdbx_reference_molecule.prd_id in the  PDB_REFERENCE_MOLECULE category.  # noqa: E501

        :param prd_id: The prd_id of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :type: str
        """

        self._prd_id = prd_id

    @property
    def source(self):
        """Gets the source of this PdbxReferenceMoleculeAnnotation.  # noqa: E501

        The source of the annoation for this entity.  # noqa: E501

        :return: The source of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :rtype: str
        """
        return self._source

    @source.setter
    def source(self, source):
        """Sets the source of this PdbxReferenceMoleculeAnnotation.

        The source of the annoation for this entity.  # noqa: E501

        :param source: The source of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :type: str
        """

        self._source = source

    @property
    def text(self):
        """Gets the text of this PdbxReferenceMoleculeAnnotation.  # noqa: E501

        Text describing the annotation for this entity.  # noqa: E501

        :return: The text of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :rtype: str
        """
        return self._text

    @text.setter
    def text(self, text):
        """Sets the text of this PdbxReferenceMoleculeAnnotation.

        Text describing the annotation for this entity.  # noqa: E501

        :param text: The text of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :type: str
        """

        self._text = text

    @property
    def type(self):
        """Gets the type of this PdbxReferenceMoleculeAnnotation.  # noqa: E501

        Type of annotation for this entity.  # noqa: E501

        :return: The type of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this PdbxReferenceMoleculeAnnotation.

        Type of annotation for this entity.  # noqa: E501

        :param type: The type of this PdbxReferenceMoleculeAnnotation.  # noqa: E501
        :type: str
        """

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
        if issubclass(PdbxReferenceMoleculeAnnotation, dict):
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
        if not isinstance(other, PdbxReferenceMoleculeAnnotation):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other