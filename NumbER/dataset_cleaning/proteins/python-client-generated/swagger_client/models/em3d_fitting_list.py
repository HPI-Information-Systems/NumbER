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

class Em3dFittingList(object):
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
        '_3d_fitting_id': 'str',
        'details': 'str',
        'id': 'str',
        'pdb_chain_id': 'str',
        'pdb_chain_residue_range': 'str',
        'pdb_entry_id': 'str'
    }

    attribute_map = {
        '_3d_fitting_id': '3d_fitting_id',
        'details': 'details',
        'id': 'id',
        'pdb_chain_id': 'pdb_chain_id',
        'pdb_chain_residue_range': 'pdb_chain_residue_range',
        'pdb_entry_id': 'pdb_entry_id'
    }

    def __init__(self, _3d_fitting_id=None, details=None, id=None, pdb_chain_id=None, pdb_chain_residue_range=None, pdb_entry_id=None):  # noqa: E501
        """Em3dFittingList - a model defined in Swagger"""  # noqa: E501
        self.__3d_fitting_id = None
        self._details = None
        self._id = None
        self._pdb_chain_id = None
        self._pdb_chain_residue_range = None
        self._pdb_entry_id = None
        self.discriminator = None
        self._3d_fitting_id = _3d_fitting_id
        if details is not None:
            self.details = details
        self.id = id
        if pdb_chain_id is not None:
            self.pdb_chain_id = pdb_chain_id
        if pdb_chain_residue_range is not None:
            self.pdb_chain_residue_range = pdb_chain_residue_range
        if pdb_entry_id is not None:
            self.pdb_entry_id = pdb_entry_id

    @property
    def _3d_fitting_id(self):
        """Gets the _3d_fitting_id of this Em3dFittingList.  # noqa: E501

        The value of _em_3d_fitting_list.3d_fitting_id is a pointer  to  _em_3d_fitting.id in the 3d_fitting category  # noqa: E501

        :return: The _3d_fitting_id of this Em3dFittingList.  # noqa: E501
        :rtype: str
        """
        return self.__3d_fitting_id

    @_3d_fitting_id.setter
    def _3d_fitting_id(self, _3d_fitting_id):
        """Sets the _3d_fitting_id of this Em3dFittingList.

        The value of _em_3d_fitting_list.3d_fitting_id is a pointer  to  _em_3d_fitting.id in the 3d_fitting category  # noqa: E501

        :param _3d_fitting_id: The _3d_fitting_id of this Em3dFittingList.  # noqa: E501
        :type: str
        """
        if _3d_fitting_id is None:
            raise ValueError("Invalid value for `_3d_fitting_id`, must not be `None`")  # noqa: E501

        self.__3d_fitting_id = _3d_fitting_id

    @property
    def details(self):
        """Gets the details of this Em3dFittingList.  # noqa: E501

        Details about the model used in fitting.  # noqa: E501

        :return: The details of this Em3dFittingList.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this Em3dFittingList.

        Details about the model used in fitting.  # noqa: E501

        :param details: The details of this Em3dFittingList.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def id(self):
        """Gets the id of this Em3dFittingList.  # noqa: E501

        PRIMARY KEY  # noqa: E501

        :return: The id of this Em3dFittingList.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Em3dFittingList.

        PRIMARY KEY  # noqa: E501

        :param id: The id of this Em3dFittingList.  # noqa: E501
        :type: str
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def pdb_chain_id(self):
        """Gets the pdb_chain_id of this Em3dFittingList.  # noqa: E501

        The ID of the biopolymer chain used for fitting, e.g., A.  Please note that only one chain can be specified per instance.  If all chains of a particular structure have been used for fitting, this field can be left blank.  # noqa: E501

        :return: The pdb_chain_id of this Em3dFittingList.  # noqa: E501
        :rtype: str
        """
        return self._pdb_chain_id

    @pdb_chain_id.setter
    def pdb_chain_id(self, pdb_chain_id):
        """Sets the pdb_chain_id of this Em3dFittingList.

        The ID of the biopolymer chain used for fitting, e.g., A.  Please note that only one chain can be specified per instance.  If all chains of a particular structure have been used for fitting, this field can be left blank.  # noqa: E501

        :param pdb_chain_id: The pdb_chain_id of this Em3dFittingList.  # noqa: E501
        :type: str
        """

        self._pdb_chain_id = pdb_chain_id

    @property
    def pdb_chain_residue_range(self):
        """Gets the pdb_chain_residue_range of this Em3dFittingList.  # noqa: E501

        Residue range for the identified chain.  # noqa: E501

        :return: The pdb_chain_residue_range of this Em3dFittingList.  # noqa: E501
        :rtype: str
        """
        return self._pdb_chain_residue_range

    @pdb_chain_residue_range.setter
    def pdb_chain_residue_range(self, pdb_chain_residue_range):
        """Sets the pdb_chain_residue_range of this Em3dFittingList.

        Residue range for the identified chain.  # noqa: E501

        :param pdb_chain_residue_range: The pdb_chain_residue_range of this Em3dFittingList.  # noqa: E501
        :type: str
        """

        self._pdb_chain_residue_range = pdb_chain_residue_range

    @property
    def pdb_entry_id(self):
        """Gets the pdb_entry_id of this Em3dFittingList.  # noqa: E501

        The PDB code for the entry used in fitting.  # noqa: E501

        :return: The pdb_entry_id of this Em3dFittingList.  # noqa: E501
        :rtype: str
        """
        return self._pdb_entry_id

    @pdb_entry_id.setter
    def pdb_entry_id(self, pdb_entry_id):
        """Sets the pdb_entry_id of this Em3dFittingList.

        The PDB code for the entry used in fitting.  # noqa: E501

        :param pdb_entry_id: The pdb_entry_id of this Em3dFittingList.  # noqa: E501
        :type: str
        """

        self._pdb_entry_id = pdb_entry_id

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
        if issubclass(Em3dFittingList, dict):
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
        if not isinstance(other, Em3dFittingList):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other