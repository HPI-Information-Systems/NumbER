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

class RefineHist(object):
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
        'cycle_id': 'str',
        'd_res_high': 'float',
        'd_res_low': 'float',
        'number_atoms_solvent': 'int',
        'number_atoms_total': 'int',
        'pdbx_b_iso_mean_ligand': 'float',
        'pdbx_b_iso_mean_solvent': 'float',
        'pdbx_number_atoms_ligand': 'int',
        'pdbx_number_atoms_nucleic_acid': 'int',
        'pdbx_number_atoms_protein': 'int',
        'pdbx_number_residues_total': 'int',
        'pdbx_refine_id': 'str'
    }

    attribute_map = {
        'cycle_id': 'cycle_id',
        'd_res_high': 'd_res_high',
        'd_res_low': 'd_res_low',
        'number_atoms_solvent': 'number_atoms_solvent',
        'number_atoms_total': 'number_atoms_total',
        'pdbx_b_iso_mean_ligand': 'pdbx_B_iso_mean_ligand',
        'pdbx_b_iso_mean_solvent': 'pdbx_B_iso_mean_solvent',
        'pdbx_number_atoms_ligand': 'pdbx_number_atoms_ligand',
        'pdbx_number_atoms_nucleic_acid': 'pdbx_number_atoms_nucleic_acid',
        'pdbx_number_atoms_protein': 'pdbx_number_atoms_protein',
        'pdbx_number_residues_total': 'pdbx_number_residues_total',
        'pdbx_refine_id': 'pdbx_refine_id'
    }

    def __init__(self, cycle_id=None, d_res_high=None, d_res_low=None, number_atoms_solvent=None, number_atoms_total=None, pdbx_b_iso_mean_ligand=None, pdbx_b_iso_mean_solvent=None, pdbx_number_atoms_ligand=None, pdbx_number_atoms_nucleic_acid=None, pdbx_number_atoms_protein=None, pdbx_number_residues_total=None, pdbx_refine_id=None):  # noqa: E501
        """RefineHist - a model defined in Swagger"""  # noqa: E501
        self._cycle_id = None
        self._d_res_high = None
        self._d_res_low = None
        self._number_atoms_solvent = None
        self._number_atoms_total = None
        self._pdbx_b_iso_mean_ligand = None
        self._pdbx_b_iso_mean_solvent = None
        self._pdbx_number_atoms_ligand = None
        self._pdbx_number_atoms_nucleic_acid = None
        self._pdbx_number_atoms_protein = None
        self._pdbx_number_residues_total = None
        self._pdbx_refine_id = None
        self.discriminator = None
        self.cycle_id = cycle_id
        if d_res_high is not None:
            self.d_res_high = d_res_high
        if d_res_low is not None:
            self.d_res_low = d_res_low
        if number_atoms_solvent is not None:
            self.number_atoms_solvent = number_atoms_solvent
        if number_atoms_total is not None:
            self.number_atoms_total = number_atoms_total
        if pdbx_b_iso_mean_ligand is not None:
            self.pdbx_b_iso_mean_ligand = pdbx_b_iso_mean_ligand
        if pdbx_b_iso_mean_solvent is not None:
            self.pdbx_b_iso_mean_solvent = pdbx_b_iso_mean_solvent
        if pdbx_number_atoms_ligand is not None:
            self.pdbx_number_atoms_ligand = pdbx_number_atoms_ligand
        if pdbx_number_atoms_nucleic_acid is not None:
            self.pdbx_number_atoms_nucleic_acid = pdbx_number_atoms_nucleic_acid
        if pdbx_number_atoms_protein is not None:
            self.pdbx_number_atoms_protein = pdbx_number_atoms_protein
        if pdbx_number_residues_total is not None:
            self.pdbx_number_residues_total = pdbx_number_residues_total
        self.pdbx_refine_id = pdbx_refine_id

    @property
    def cycle_id(self):
        """Gets the cycle_id of this RefineHist.  # noqa: E501

        The value of _refine_hist.cycle_id must uniquely identify a  record in the REFINE_HIST list.   Note that this item need not be a number; it can be any unique  identifier.  # noqa: E501

        :return: The cycle_id of this RefineHist.  # noqa: E501
        :rtype: str
        """
        return self._cycle_id

    @cycle_id.setter
    def cycle_id(self, cycle_id):
        """Sets the cycle_id of this RefineHist.

        The value of _refine_hist.cycle_id must uniquely identify a  record in the REFINE_HIST list.   Note that this item need not be a number; it can be any unique  identifier.  # noqa: E501

        :param cycle_id: The cycle_id of this RefineHist.  # noqa: E501
        :type: str
        """
        if cycle_id is None:
            raise ValueError("Invalid value for `cycle_id`, must not be `None`")  # noqa: E501

        self._cycle_id = cycle_id

    @property
    def d_res_high(self):
        """Gets the d_res_high of this RefineHist.  # noqa: E501

        The lowest value for the interplanar spacings for the  reflection data for this cycle of refinement. This is called  the highest resolution.  # noqa: E501

        :return: The d_res_high of this RefineHist.  # noqa: E501
        :rtype: float
        """
        return self._d_res_high

    @d_res_high.setter
    def d_res_high(self, d_res_high):
        """Sets the d_res_high of this RefineHist.

        The lowest value for the interplanar spacings for the  reflection data for this cycle of refinement. This is called  the highest resolution.  # noqa: E501

        :param d_res_high: The d_res_high of this RefineHist.  # noqa: E501
        :type: float
        """

        self._d_res_high = d_res_high

    @property
    def d_res_low(self):
        """Gets the d_res_low of this RefineHist.  # noqa: E501

        The highest value for the interplanar spacings for the  reflection data for this cycle of refinement. This is  called the lowest resolution.  # noqa: E501

        :return: The d_res_low of this RefineHist.  # noqa: E501
        :rtype: float
        """
        return self._d_res_low

    @d_res_low.setter
    def d_res_low(self, d_res_low):
        """Sets the d_res_low of this RefineHist.

        The highest value for the interplanar spacings for the  reflection data for this cycle of refinement. This is  called the lowest resolution.  # noqa: E501

        :param d_res_low: The d_res_low of this RefineHist.  # noqa: E501
        :type: float
        """

        self._d_res_low = d_res_low

    @property
    def number_atoms_solvent(self):
        """Gets the number_atoms_solvent of this RefineHist.  # noqa: E501

        The number of solvent atoms that were included in the model at  this cycle of the refinement.  # noqa: E501

        :return: The number_atoms_solvent of this RefineHist.  # noqa: E501
        :rtype: int
        """
        return self._number_atoms_solvent

    @number_atoms_solvent.setter
    def number_atoms_solvent(self, number_atoms_solvent):
        """Sets the number_atoms_solvent of this RefineHist.

        The number of solvent atoms that were included in the model at  this cycle of the refinement.  # noqa: E501

        :param number_atoms_solvent: The number_atoms_solvent of this RefineHist.  # noqa: E501
        :type: int
        """

        self._number_atoms_solvent = number_atoms_solvent

    @property
    def number_atoms_total(self):
        """Gets the number_atoms_total of this RefineHist.  # noqa: E501

        The total number of atoms that were included in the model at  this cycle of the refinement.  # noqa: E501

        :return: The number_atoms_total of this RefineHist.  # noqa: E501
        :rtype: int
        """
        return self._number_atoms_total

    @number_atoms_total.setter
    def number_atoms_total(self, number_atoms_total):
        """Sets the number_atoms_total of this RefineHist.

        The total number of atoms that were included in the model at  this cycle of the refinement.  # noqa: E501

        :param number_atoms_total: The number_atoms_total of this RefineHist.  # noqa: E501
        :type: int
        """

        self._number_atoms_total = number_atoms_total

    @property
    def pdbx_b_iso_mean_ligand(self):
        """Gets the pdbx_b_iso_mean_ligand of this RefineHist.  # noqa: E501

        Mean isotropic B-value for ligand molecules included in refinement.  # noqa: E501

        :return: The pdbx_b_iso_mean_ligand of this RefineHist.  # noqa: E501
        :rtype: float
        """
        return self._pdbx_b_iso_mean_ligand

    @pdbx_b_iso_mean_ligand.setter
    def pdbx_b_iso_mean_ligand(self, pdbx_b_iso_mean_ligand):
        """Sets the pdbx_b_iso_mean_ligand of this RefineHist.

        Mean isotropic B-value for ligand molecules included in refinement.  # noqa: E501

        :param pdbx_b_iso_mean_ligand: The pdbx_b_iso_mean_ligand of this RefineHist.  # noqa: E501
        :type: float
        """

        self._pdbx_b_iso_mean_ligand = pdbx_b_iso_mean_ligand

    @property
    def pdbx_b_iso_mean_solvent(self):
        """Gets the pdbx_b_iso_mean_solvent of this RefineHist.  # noqa: E501

        Mean isotropic B-value for solvent molecules included in refinement.  # noqa: E501

        :return: The pdbx_b_iso_mean_solvent of this RefineHist.  # noqa: E501
        :rtype: float
        """
        return self._pdbx_b_iso_mean_solvent

    @pdbx_b_iso_mean_solvent.setter
    def pdbx_b_iso_mean_solvent(self, pdbx_b_iso_mean_solvent):
        """Sets the pdbx_b_iso_mean_solvent of this RefineHist.

        Mean isotropic B-value for solvent molecules included in refinement.  # noqa: E501

        :param pdbx_b_iso_mean_solvent: The pdbx_b_iso_mean_solvent of this RefineHist.  # noqa: E501
        :type: float
        """

        self._pdbx_b_iso_mean_solvent = pdbx_b_iso_mean_solvent

    @property
    def pdbx_number_atoms_ligand(self):
        """Gets the pdbx_number_atoms_ligand of this RefineHist.  # noqa: E501

        Number of ligand atoms included in refinement  # noqa: E501

        :return: The pdbx_number_atoms_ligand of this RefineHist.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_number_atoms_ligand

    @pdbx_number_atoms_ligand.setter
    def pdbx_number_atoms_ligand(self, pdbx_number_atoms_ligand):
        """Sets the pdbx_number_atoms_ligand of this RefineHist.

        Number of ligand atoms included in refinement  # noqa: E501

        :param pdbx_number_atoms_ligand: The pdbx_number_atoms_ligand of this RefineHist.  # noqa: E501
        :type: int
        """

        self._pdbx_number_atoms_ligand = pdbx_number_atoms_ligand

    @property
    def pdbx_number_atoms_nucleic_acid(self):
        """Gets the pdbx_number_atoms_nucleic_acid of this RefineHist.  # noqa: E501

        Number of nucleic atoms included in refinement  # noqa: E501

        :return: The pdbx_number_atoms_nucleic_acid of this RefineHist.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_number_atoms_nucleic_acid

    @pdbx_number_atoms_nucleic_acid.setter
    def pdbx_number_atoms_nucleic_acid(self, pdbx_number_atoms_nucleic_acid):
        """Sets the pdbx_number_atoms_nucleic_acid of this RefineHist.

        Number of nucleic atoms included in refinement  # noqa: E501

        :param pdbx_number_atoms_nucleic_acid: The pdbx_number_atoms_nucleic_acid of this RefineHist.  # noqa: E501
        :type: int
        """

        self._pdbx_number_atoms_nucleic_acid = pdbx_number_atoms_nucleic_acid

    @property
    def pdbx_number_atoms_protein(self):
        """Gets the pdbx_number_atoms_protein of this RefineHist.  # noqa: E501

        Number of protein atoms included in refinement  # noqa: E501

        :return: The pdbx_number_atoms_protein of this RefineHist.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_number_atoms_protein

    @pdbx_number_atoms_protein.setter
    def pdbx_number_atoms_protein(self, pdbx_number_atoms_protein):
        """Sets the pdbx_number_atoms_protein of this RefineHist.

        Number of protein atoms included in refinement  # noqa: E501

        :param pdbx_number_atoms_protein: The pdbx_number_atoms_protein of this RefineHist.  # noqa: E501
        :type: int
        """

        self._pdbx_number_atoms_protein = pdbx_number_atoms_protein

    @property
    def pdbx_number_residues_total(self):
        """Gets the pdbx_number_residues_total of this RefineHist.  # noqa: E501

        Total number of polymer residues included in refinement.  # noqa: E501

        :return: The pdbx_number_residues_total of this RefineHist.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_number_residues_total

    @pdbx_number_residues_total.setter
    def pdbx_number_residues_total(self, pdbx_number_residues_total):
        """Sets the pdbx_number_residues_total of this RefineHist.

        Total number of polymer residues included in refinement.  # noqa: E501

        :param pdbx_number_residues_total: The pdbx_number_residues_total of this RefineHist.  # noqa: E501
        :type: int
        """

        self._pdbx_number_residues_total = pdbx_number_residues_total

    @property
    def pdbx_refine_id(self):
        """Gets the pdbx_refine_id of this RefineHist.  # noqa: E501

        This data item uniquely identifies a refinement within an entry.  _refine_hist.pdbx_refine_id can be used to distinguish the results  of joint refinements.  # noqa: E501

        :return: The pdbx_refine_id of this RefineHist.  # noqa: E501
        :rtype: str
        """
        return self._pdbx_refine_id

    @pdbx_refine_id.setter
    def pdbx_refine_id(self, pdbx_refine_id):
        """Sets the pdbx_refine_id of this RefineHist.

        This data item uniquely identifies a refinement within an entry.  _refine_hist.pdbx_refine_id can be used to distinguish the results  of joint refinements.  # noqa: E501

        :param pdbx_refine_id: The pdbx_refine_id of this RefineHist.  # noqa: E501
        :type: str
        """
        if pdbx_refine_id is None:
            raise ValueError("Invalid value for `pdbx_refine_id`, must not be `None`")  # noqa: E501

        self._pdbx_refine_id = pdbx_refine_id

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
        if issubclass(RefineHist, dict):
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
        if not isinstance(other, RefineHist):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other