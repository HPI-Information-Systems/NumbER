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

class RcsbPolymerStructConnConnectPartner(object):
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
        'label_alt_id': 'str',
        'label_asym_id': 'str',
        'label_atom_id': 'str',
        'label_comp_id': 'str',
        'label_seq_id': 'int',
        'symmetry': 'str'
    }

    attribute_map = {
        'label_alt_id': 'label_alt_id',
        'label_asym_id': 'label_asym_id',
        'label_atom_id': 'label_atom_id',
        'label_comp_id': 'label_comp_id',
        'label_seq_id': 'label_seq_id',
        'symmetry': 'symmetry'
    }

    def __init__(self, label_alt_id=None, label_asym_id=None, label_atom_id=None, label_comp_id=None, label_seq_id=None, symmetry=None):  # noqa: E501
        """RcsbPolymerStructConnConnectPartner - a model defined in Swagger"""  # noqa: E501
        self._label_alt_id = None
        self._label_asym_id = None
        self._label_atom_id = None
        self._label_comp_id = None
        self._label_seq_id = None
        self._symmetry = None
        self.discriminator = None
        if label_alt_id is not None:
            self.label_alt_id = label_alt_id
        self.label_asym_id = label_asym_id
        if label_atom_id is not None:
            self.label_atom_id = label_atom_id
        self.label_comp_id = label_comp_id
        if label_seq_id is not None:
            self.label_seq_id = label_seq_id
        if symmetry is not None:
            self.symmetry = symmetry

    @property
    def label_alt_id(self):
        """Gets the label_alt_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_alt_id in the  ATOM_SITE category.  # noqa: E501

        :return: The label_alt_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :rtype: str
        """
        return self._label_alt_id

    @label_alt_id.setter
    def label_alt_id(self, label_alt_id):
        """Sets the label_alt_id of this RcsbPolymerStructConnConnectPartner.

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_alt_id in the  ATOM_SITE category.  # noqa: E501

        :param label_alt_id: The label_alt_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :type: str
        """

        self._label_alt_id = label_alt_id

    @property
    def label_asym_id(self):
        """Gets the label_asym_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_asym_id in the  ATOM_SITE category.  # noqa: E501

        :return: The label_asym_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :rtype: str
        """
        return self._label_asym_id

    @label_asym_id.setter
    def label_asym_id(self, label_asym_id):
        """Sets the label_asym_id of this RcsbPolymerStructConnConnectPartner.

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_asym_id in the  ATOM_SITE category.  # noqa: E501

        :param label_asym_id: The label_asym_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :type: str
        """
        if label_asym_id is None:
            raise ValueError("Invalid value for `label_asym_id`, must not be `None`")  # noqa: E501

        self._label_asym_id = label_asym_id

    @property
    def label_atom_id(self):
        """Gets the label_atom_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _chem_comp_atom.atom_id in the  CHEM_COMP_ATOM category.  # noqa: E501

        :return: The label_atom_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :rtype: str
        """
        return self._label_atom_id

    @label_atom_id.setter
    def label_atom_id(self, label_atom_id):
        """Sets the label_atom_id of this RcsbPolymerStructConnConnectPartner.

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _chem_comp_atom.atom_id in the  CHEM_COMP_ATOM category.  # noqa: E501

        :param label_atom_id: The label_atom_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :type: str
        """

        self._label_atom_id = label_atom_id

    @property
    def label_comp_id(self):
        """Gets the label_comp_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_comp_id in the  ATOM_SITE category.  # noqa: E501

        :return: The label_comp_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :rtype: str
        """
        return self._label_comp_id

    @label_comp_id.setter
    def label_comp_id(self, label_comp_id):
        """Sets the label_comp_id of this RcsbPolymerStructConnConnectPartner.

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_comp_id in the  ATOM_SITE category.  # noqa: E501

        :param label_comp_id: The label_comp_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :type: str
        """
        if label_comp_id is None:
            raise ValueError("Invalid value for `label_comp_id`, must not be `None`")  # noqa: E501

        self._label_comp_id = label_comp_id

    @property
    def label_seq_id(self):
        """Gets the label_seq_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_seq_id in the  ATOM_SITE category.  # noqa: E501

        :return: The label_seq_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :rtype: int
        """
        return self._label_seq_id

    @label_seq_id.setter
    def label_seq_id(self, label_seq_id):
        """Sets the label_seq_id of this RcsbPolymerStructConnConnectPartner.

        A component of the identifier for the partner in the structure  connection.   This data item is a pointer to _atom_site.label_seq_id in the  ATOM_SITE category.  # noqa: E501

        :param label_seq_id: The label_seq_id of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :type: int
        """

        self._label_seq_id = label_seq_id

    @property
    def symmetry(self):
        """Gets the symmetry of this RcsbPolymerStructConnConnectPartner.  # noqa: E501

        Describes the symmetry operation that should be applied to the  atom set specified by _rcsb_polymer_struct_conn.connect_partner_label* to generate the  partner in the structure connection.  # noqa: E501

        :return: The symmetry of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :rtype: str
        """
        return self._symmetry

    @symmetry.setter
    def symmetry(self, symmetry):
        """Sets the symmetry of this RcsbPolymerStructConnConnectPartner.

        Describes the symmetry operation that should be applied to the  atom set specified by _rcsb_polymer_struct_conn.connect_partner_label* to generate the  partner in the structure connection.  # noqa: E501

        :param symmetry: The symmetry of this RcsbPolymerStructConnConnectPartner.  # noqa: E501
        :type: str
        """

        self._symmetry = symmetry

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
        if issubclass(RcsbPolymerStructConnConnectPartner, dict):
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
        if not isinstance(other, RcsbPolymerStructConnConnectPartner):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
