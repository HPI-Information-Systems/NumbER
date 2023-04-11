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

class PdbxReferenceMoleculeRelatedStructures(object):
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
        'citation_id': 'str',
        'db_accession': 'str',
        'db_code': 'str',
        'db_name': 'str',
        'family_prd_id': 'str',
        'formula': 'str',
        'name': 'str',
        'ordinal': 'int'
    }

    attribute_map = {
        'citation_id': 'citation_id',
        'db_accession': 'db_accession',
        'db_code': 'db_code',
        'db_name': 'db_name',
        'family_prd_id': 'family_prd_id',
        'formula': 'formula',
        'name': 'name',
        'ordinal': 'ordinal'
    }

    def __init__(self, citation_id=None, db_accession=None, db_code=None, db_name=None, family_prd_id=None, formula=None, name=None, ordinal=None):  # noqa: E501
        """PdbxReferenceMoleculeRelatedStructures - a model defined in Swagger"""  # noqa: E501
        self._citation_id = None
        self._db_accession = None
        self._db_code = None
        self._db_name = None
        self._family_prd_id = None
        self._formula = None
        self._name = None
        self._ordinal = None
        self.discriminator = None
        if citation_id is not None:
            self.citation_id = citation_id
        if db_accession is not None:
            self.db_accession = db_accession
        if db_code is not None:
            self.db_code = db_code
        if db_name is not None:
            self.db_name = db_name
        self.family_prd_id = family_prd_id
        if formula is not None:
            self.formula = formula
        if name is not None:
            self.name = name
        self.ordinal = ordinal

    @property
    def citation_id(self):
        """Gets the citation_id of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        A link to related reference information in the citation category.  # noqa: E501

        :return: The citation_id of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: str
        """
        return self._citation_id

    @citation_id.setter
    def citation_id(self, citation_id):
        """Sets the citation_id of this PdbxReferenceMoleculeRelatedStructures.

        A link to related reference information in the citation category.  # noqa: E501

        :param citation_id: The citation_id of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: str
        """

        self._citation_id = citation_id

    @property
    def db_accession(self):
        """Gets the db_accession of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        The database accession code for the related structure reference.  # noqa: E501

        :return: The db_accession of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: str
        """
        return self._db_accession

    @db_accession.setter
    def db_accession(self, db_accession):
        """Sets the db_accession of this PdbxReferenceMoleculeRelatedStructures.

        The database accession code for the related structure reference.  # noqa: E501

        :param db_accession: The db_accession of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: str
        """

        self._db_accession = db_accession

    @property
    def db_code(self):
        """Gets the db_code of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        The database identifier code for the related structure reference.  # noqa: E501

        :return: The db_code of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: str
        """
        return self._db_code

    @db_code.setter
    def db_code(self, db_code):
        """Sets the db_code of this PdbxReferenceMoleculeRelatedStructures.

        The database identifier code for the related structure reference.  # noqa: E501

        :param db_code: The db_code of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: str
        """

        self._db_code = db_code

    @property
    def db_name(self):
        """Gets the db_name of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        The database name for the related structure reference.  # noqa: E501

        :return: The db_name of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: str
        """
        return self._db_name

    @db_name.setter
    def db_name(self, db_name):
        """Sets the db_name of this PdbxReferenceMoleculeRelatedStructures.

        The database name for the related structure reference.  # noqa: E501

        :param db_name: The db_name of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: str
        """

        self._db_name = db_name

    @property
    def family_prd_id(self):
        """Gets the family_prd_id of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        The value of _pdbx_reference_molecule_related_structures.family_prd_id is a reference to  _pdbx_reference_molecule_list.family_prd_id in category PDBX_REFERENCE_MOLECULE_FAMILY_LIST.  # noqa: E501

        :return: The family_prd_id of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: str
        """
        return self._family_prd_id

    @family_prd_id.setter
    def family_prd_id(self, family_prd_id):
        """Sets the family_prd_id of this PdbxReferenceMoleculeRelatedStructures.

        The value of _pdbx_reference_molecule_related_structures.family_prd_id is a reference to  _pdbx_reference_molecule_list.family_prd_id in category PDBX_REFERENCE_MOLECULE_FAMILY_LIST.  # noqa: E501

        :param family_prd_id: The family_prd_id of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: str
        """
        if family_prd_id is None:
            raise ValueError("Invalid value for `family_prd_id`, must not be `None`")  # noqa: E501

        self._family_prd_id = family_prd_id

    @property
    def formula(self):
        """Gets the formula of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        The formula for the reference entity. Formulae are written  according to the rules:   1. Only recognised element symbols may be used.   2. Each element symbol is followed by a 'count' number. A count     of '1' may be omitted.   3. A space or parenthesis must separate each element symbol and     its count, but in general parentheses are not used.   4. The order of elements depends on whether or not carbon is     present. If carbon is present, the order should be: C, then     H, then the other elements in alphabetical order of their     symbol. If carbon is not present, the elements are listed     purely in alphabetic order of their symbol. This is the     'Hill' system used by Chemical Abstracts.  # noqa: E501

        :return: The formula of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: str
        """
        return self._formula

    @formula.setter
    def formula(self, formula):
        """Sets the formula of this PdbxReferenceMoleculeRelatedStructures.

        The formula for the reference entity. Formulae are written  according to the rules:   1. Only recognised element symbols may be used.   2. Each element symbol is followed by a 'count' number. A count     of '1' may be omitted.   3. A space or parenthesis must separate each element symbol and     its count, but in general parentheses are not used.   4. The order of elements depends on whether or not carbon is     present. If carbon is present, the order should be: C, then     H, then the other elements in alphabetical order of their     symbol. If carbon is not present, the elements are listed     purely in alphabetic order of their symbol. This is the     'Hill' system used by Chemical Abstracts.  # noqa: E501

        :param formula: The formula of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: str
        """

        self._formula = formula

    @property
    def name(self):
        """Gets the name of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        The chemical name for the structure entry in the related database  # noqa: E501

        :return: The name of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this PdbxReferenceMoleculeRelatedStructures.

        The chemical name for the structure entry in the related database  # noqa: E501

        :param name: The name of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def ordinal(self):
        """Gets the ordinal of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501

        The value of _pdbx_reference_molecule_related_structures.ordinal distinguishes  related structural data for each entity.  # noqa: E501

        :return: The ordinal of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :rtype: int
        """
        return self._ordinal

    @ordinal.setter
    def ordinal(self, ordinal):
        """Sets the ordinal of this PdbxReferenceMoleculeRelatedStructures.

        The value of _pdbx_reference_molecule_related_structures.ordinal distinguishes  related structural data for each entity.  # noqa: E501

        :param ordinal: The ordinal of this PdbxReferenceMoleculeRelatedStructures.  # noqa: E501
        :type: int
        """
        if ordinal is None:
            raise ValueError("Invalid value for `ordinal`, must not be `None`")  # noqa: E501

        self._ordinal = ordinal

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
        if issubclass(PdbxReferenceMoleculeRelatedStructures, dict):
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
        if not isinstance(other, PdbxReferenceMoleculeRelatedStructures):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
