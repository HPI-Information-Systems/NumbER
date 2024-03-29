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

class PdbxNmrEnsemble(object):
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
        'average_constraint_violations_per_residue': 'int',
        'average_constraints_per_residue': 'int',
        'average_distance_constraint_violation': 'float',
        'average_torsion_angle_constraint_violation': 'float',
        'conformer_selection_criteria': 'str',
        'conformers_calculated_total_number': 'int',
        'conformers_submitted_total_number': 'int',
        'distance_constraint_violation_method': 'str',
        'maximum_distance_constraint_violation': 'float',
        'maximum_lower_distance_constraint_violation': 'float',
        'maximum_torsion_angle_constraint_violation': 'float',
        'maximum_upper_distance_constraint_violation': 'float',
        'representative_conformer': 'int',
        'torsion_angle_constraint_violation_method': 'str'
    }

    attribute_map = {
        'average_constraint_violations_per_residue': 'average_constraint_violations_per_residue',
        'average_constraints_per_residue': 'average_constraints_per_residue',
        'average_distance_constraint_violation': 'average_distance_constraint_violation',
        'average_torsion_angle_constraint_violation': 'average_torsion_angle_constraint_violation',
        'conformer_selection_criteria': 'conformer_selection_criteria',
        'conformers_calculated_total_number': 'conformers_calculated_total_number',
        'conformers_submitted_total_number': 'conformers_submitted_total_number',
        'distance_constraint_violation_method': 'distance_constraint_violation_method',
        'maximum_distance_constraint_violation': 'maximum_distance_constraint_violation',
        'maximum_lower_distance_constraint_violation': 'maximum_lower_distance_constraint_violation',
        'maximum_torsion_angle_constraint_violation': 'maximum_torsion_angle_constraint_violation',
        'maximum_upper_distance_constraint_violation': 'maximum_upper_distance_constraint_violation',
        'representative_conformer': 'representative_conformer',
        'torsion_angle_constraint_violation_method': 'torsion_angle_constraint_violation_method'
    }

    def __init__(self, average_constraint_violations_per_residue=None, average_constraints_per_residue=None, average_distance_constraint_violation=None, average_torsion_angle_constraint_violation=None, conformer_selection_criteria=None, conformers_calculated_total_number=None, conformers_submitted_total_number=None, distance_constraint_violation_method=None, maximum_distance_constraint_violation=None, maximum_lower_distance_constraint_violation=None, maximum_torsion_angle_constraint_violation=None, maximum_upper_distance_constraint_violation=None, representative_conformer=None, torsion_angle_constraint_violation_method=None):  # noqa: E501
        """PdbxNmrEnsemble - a model defined in Swagger"""  # noqa: E501
        self._average_constraint_violations_per_residue = None
        self._average_constraints_per_residue = None
        self._average_distance_constraint_violation = None
        self._average_torsion_angle_constraint_violation = None
        self._conformer_selection_criteria = None
        self._conformers_calculated_total_number = None
        self._conformers_submitted_total_number = None
        self._distance_constraint_violation_method = None
        self._maximum_distance_constraint_violation = None
        self._maximum_lower_distance_constraint_violation = None
        self._maximum_torsion_angle_constraint_violation = None
        self._maximum_upper_distance_constraint_violation = None
        self._representative_conformer = None
        self._torsion_angle_constraint_violation_method = None
        self.discriminator = None
        if average_constraint_violations_per_residue is not None:
            self.average_constraint_violations_per_residue = average_constraint_violations_per_residue
        if average_constraints_per_residue is not None:
            self.average_constraints_per_residue = average_constraints_per_residue
        if average_distance_constraint_violation is not None:
            self.average_distance_constraint_violation = average_distance_constraint_violation
        if average_torsion_angle_constraint_violation is not None:
            self.average_torsion_angle_constraint_violation = average_torsion_angle_constraint_violation
        if conformer_selection_criteria is not None:
            self.conformer_selection_criteria = conformer_selection_criteria
        if conformers_calculated_total_number is not None:
            self.conformers_calculated_total_number = conformers_calculated_total_number
        if conformers_submitted_total_number is not None:
            self.conformers_submitted_total_number = conformers_submitted_total_number
        if distance_constraint_violation_method is not None:
            self.distance_constraint_violation_method = distance_constraint_violation_method
        if maximum_distance_constraint_violation is not None:
            self.maximum_distance_constraint_violation = maximum_distance_constraint_violation
        if maximum_lower_distance_constraint_violation is not None:
            self.maximum_lower_distance_constraint_violation = maximum_lower_distance_constraint_violation
        if maximum_torsion_angle_constraint_violation is not None:
            self.maximum_torsion_angle_constraint_violation = maximum_torsion_angle_constraint_violation
        if maximum_upper_distance_constraint_violation is not None:
            self.maximum_upper_distance_constraint_violation = maximum_upper_distance_constraint_violation
        if representative_conformer is not None:
            self.representative_conformer = representative_conformer
        if torsion_angle_constraint_violation_method is not None:
            self.torsion_angle_constraint_violation_method = torsion_angle_constraint_violation_method

    @property
    def average_constraint_violations_per_residue(self):
        """Gets the average_constraint_violations_per_residue of this PdbxNmrEnsemble.  # noqa: E501

        The average number of constraint violations on a per residue basis for  the ensemble.  # noqa: E501

        :return: The average_constraint_violations_per_residue of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: int
        """
        return self._average_constraint_violations_per_residue

    @average_constraint_violations_per_residue.setter
    def average_constraint_violations_per_residue(self, average_constraint_violations_per_residue):
        """Sets the average_constraint_violations_per_residue of this PdbxNmrEnsemble.

        The average number of constraint violations on a per residue basis for  the ensemble.  # noqa: E501

        :param average_constraint_violations_per_residue: The average_constraint_violations_per_residue of this PdbxNmrEnsemble.  # noqa: E501
        :type: int
        """

        self._average_constraint_violations_per_residue = average_constraint_violations_per_residue

    @property
    def average_constraints_per_residue(self):
        """Gets the average_constraints_per_residue of this PdbxNmrEnsemble.  # noqa: E501

        The average number of constraints per residue for the ensemble  # noqa: E501

        :return: The average_constraints_per_residue of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: int
        """
        return self._average_constraints_per_residue

    @average_constraints_per_residue.setter
    def average_constraints_per_residue(self, average_constraints_per_residue):
        """Sets the average_constraints_per_residue of this PdbxNmrEnsemble.

        The average number of constraints per residue for the ensemble  # noqa: E501

        :param average_constraints_per_residue: The average_constraints_per_residue of this PdbxNmrEnsemble.  # noqa: E501
        :type: int
        """

        self._average_constraints_per_residue = average_constraints_per_residue

    @property
    def average_distance_constraint_violation(self):
        """Gets the average_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501

        The average distance restraint violation for the ensemble.  # noqa: E501

        :return: The average_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: float
        """
        return self._average_distance_constraint_violation

    @average_distance_constraint_violation.setter
    def average_distance_constraint_violation(self, average_distance_constraint_violation):
        """Sets the average_distance_constraint_violation of this PdbxNmrEnsemble.

        The average distance restraint violation for the ensemble.  # noqa: E501

        :param average_distance_constraint_violation: The average_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :type: float
        """

        self._average_distance_constraint_violation = average_distance_constraint_violation

    @property
    def average_torsion_angle_constraint_violation(self):
        """Gets the average_torsion_angle_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501

        The average torsion angle constraint violation for the ensemble.  # noqa: E501

        :return: The average_torsion_angle_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: float
        """
        return self._average_torsion_angle_constraint_violation

    @average_torsion_angle_constraint_violation.setter
    def average_torsion_angle_constraint_violation(self, average_torsion_angle_constraint_violation):
        """Sets the average_torsion_angle_constraint_violation of this PdbxNmrEnsemble.

        The average torsion angle constraint violation for the ensemble.  # noqa: E501

        :param average_torsion_angle_constraint_violation: The average_torsion_angle_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :type: float
        """

        self._average_torsion_angle_constraint_violation = average_torsion_angle_constraint_violation

    @property
    def conformer_selection_criteria(self):
        """Gets the conformer_selection_criteria of this PdbxNmrEnsemble.  # noqa: E501

        By highlighting the appropriate choice(s), describe how the submitted conformer (models) were selected.  # noqa: E501

        :return: The conformer_selection_criteria of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: str
        """
        return self._conformer_selection_criteria

    @conformer_selection_criteria.setter
    def conformer_selection_criteria(self, conformer_selection_criteria):
        """Sets the conformer_selection_criteria of this PdbxNmrEnsemble.

        By highlighting the appropriate choice(s), describe how the submitted conformer (models) were selected.  # noqa: E501

        :param conformer_selection_criteria: The conformer_selection_criteria of this PdbxNmrEnsemble.  # noqa: E501
        :type: str
        """

        self._conformer_selection_criteria = conformer_selection_criteria

    @property
    def conformers_calculated_total_number(self):
        """Gets the conformers_calculated_total_number of this PdbxNmrEnsemble.  # noqa: E501

        The total number of conformer (models) that were calculated in the final round.  # noqa: E501

        :return: The conformers_calculated_total_number of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: int
        """
        return self._conformers_calculated_total_number

    @conformers_calculated_total_number.setter
    def conformers_calculated_total_number(self, conformers_calculated_total_number):
        """Sets the conformers_calculated_total_number of this PdbxNmrEnsemble.

        The total number of conformer (models) that were calculated in the final round.  # noqa: E501

        :param conformers_calculated_total_number: The conformers_calculated_total_number of this PdbxNmrEnsemble.  # noqa: E501
        :type: int
        """

        self._conformers_calculated_total_number = conformers_calculated_total_number

    @property
    def conformers_submitted_total_number(self):
        """Gets the conformers_submitted_total_number of this PdbxNmrEnsemble.  # noqa: E501

        The number of conformer (models) that are submitted for the ensemble.  # noqa: E501

        :return: The conformers_submitted_total_number of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: int
        """
        return self._conformers_submitted_total_number

    @conformers_submitted_total_number.setter
    def conformers_submitted_total_number(self, conformers_submitted_total_number):
        """Sets the conformers_submitted_total_number of this PdbxNmrEnsemble.

        The number of conformer (models) that are submitted for the ensemble.  # noqa: E501

        :param conformers_submitted_total_number: The conformers_submitted_total_number of this PdbxNmrEnsemble.  # noqa: E501
        :type: int
        """

        self._conformers_submitted_total_number = conformers_submitted_total_number

    @property
    def distance_constraint_violation_method(self):
        """Gets the distance_constraint_violation_method of this PdbxNmrEnsemble.  # noqa: E501

        Describe the method used to calculate the distance constraint violation statistics,  i.e. are they calculated over all the distance constraints or calculated for  violations only?  # noqa: E501

        :return: The distance_constraint_violation_method of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: str
        """
        return self._distance_constraint_violation_method

    @distance_constraint_violation_method.setter
    def distance_constraint_violation_method(self, distance_constraint_violation_method):
        """Sets the distance_constraint_violation_method of this PdbxNmrEnsemble.

        Describe the method used to calculate the distance constraint violation statistics,  i.e. are they calculated over all the distance constraints or calculated for  violations only?  # noqa: E501

        :param distance_constraint_violation_method: The distance_constraint_violation_method of this PdbxNmrEnsemble.  # noqa: E501
        :type: str
        """

        self._distance_constraint_violation_method = distance_constraint_violation_method

    @property
    def maximum_distance_constraint_violation(self):
        """Gets the maximum_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501

        The maximum distance constraint violation for the ensemble.  # noqa: E501

        :return: The maximum_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: float
        """
        return self._maximum_distance_constraint_violation

    @maximum_distance_constraint_violation.setter
    def maximum_distance_constraint_violation(self, maximum_distance_constraint_violation):
        """Sets the maximum_distance_constraint_violation of this PdbxNmrEnsemble.

        The maximum distance constraint violation for the ensemble.  # noqa: E501

        :param maximum_distance_constraint_violation: The maximum_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :type: float
        """

        self._maximum_distance_constraint_violation = maximum_distance_constraint_violation

    @property
    def maximum_lower_distance_constraint_violation(self):
        """Gets the maximum_lower_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501

        The maximum lower distance constraint violation for the ensemble.  # noqa: E501

        :return: The maximum_lower_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: float
        """
        return self._maximum_lower_distance_constraint_violation

    @maximum_lower_distance_constraint_violation.setter
    def maximum_lower_distance_constraint_violation(self, maximum_lower_distance_constraint_violation):
        """Sets the maximum_lower_distance_constraint_violation of this PdbxNmrEnsemble.

        The maximum lower distance constraint violation for the ensemble.  # noqa: E501

        :param maximum_lower_distance_constraint_violation: The maximum_lower_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :type: float
        """

        self._maximum_lower_distance_constraint_violation = maximum_lower_distance_constraint_violation

    @property
    def maximum_torsion_angle_constraint_violation(self):
        """Gets the maximum_torsion_angle_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501

        The maximum torsion angle constraint violation for the ensemble.  # noqa: E501

        :return: The maximum_torsion_angle_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: float
        """
        return self._maximum_torsion_angle_constraint_violation

    @maximum_torsion_angle_constraint_violation.setter
    def maximum_torsion_angle_constraint_violation(self, maximum_torsion_angle_constraint_violation):
        """Sets the maximum_torsion_angle_constraint_violation of this PdbxNmrEnsemble.

        The maximum torsion angle constraint violation for the ensemble.  # noqa: E501

        :param maximum_torsion_angle_constraint_violation: The maximum_torsion_angle_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :type: float
        """

        self._maximum_torsion_angle_constraint_violation = maximum_torsion_angle_constraint_violation

    @property
    def maximum_upper_distance_constraint_violation(self):
        """Gets the maximum_upper_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501

        The maximum upper distance constraint violation for the ensemble.  # noqa: E501

        :return: The maximum_upper_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: float
        """
        return self._maximum_upper_distance_constraint_violation

    @maximum_upper_distance_constraint_violation.setter
    def maximum_upper_distance_constraint_violation(self, maximum_upper_distance_constraint_violation):
        """Sets the maximum_upper_distance_constraint_violation of this PdbxNmrEnsemble.

        The maximum upper distance constraint violation for the ensemble.  # noqa: E501

        :param maximum_upper_distance_constraint_violation: The maximum_upper_distance_constraint_violation of this PdbxNmrEnsemble.  # noqa: E501
        :type: float
        """

        self._maximum_upper_distance_constraint_violation = maximum_upper_distance_constraint_violation

    @property
    def representative_conformer(self):
        """Gets the representative_conformer of this PdbxNmrEnsemble.  # noqa: E501

        The number of the conformer identified as most representative.  # noqa: E501

        :return: The representative_conformer of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: int
        """
        return self._representative_conformer

    @representative_conformer.setter
    def representative_conformer(self, representative_conformer):
        """Sets the representative_conformer of this PdbxNmrEnsemble.

        The number of the conformer identified as most representative.  # noqa: E501

        :param representative_conformer: The representative_conformer of this PdbxNmrEnsemble.  # noqa: E501
        :type: int
        """

        self._representative_conformer = representative_conformer

    @property
    def torsion_angle_constraint_violation_method(self):
        """Gets the torsion_angle_constraint_violation_method of this PdbxNmrEnsemble.  # noqa: E501

        This item describes the method used to calculate the torsion angle constraint violation statistics. i.e. are the entered values based on all torsion angle or calculated for violations only?  # noqa: E501

        :return: The torsion_angle_constraint_violation_method of this PdbxNmrEnsemble.  # noqa: E501
        :rtype: str
        """
        return self._torsion_angle_constraint_violation_method

    @torsion_angle_constraint_violation_method.setter
    def torsion_angle_constraint_violation_method(self, torsion_angle_constraint_violation_method):
        """Sets the torsion_angle_constraint_violation_method of this PdbxNmrEnsemble.

        This item describes the method used to calculate the torsion angle constraint violation statistics. i.e. are the entered values based on all torsion angle or calculated for violations only?  # noqa: E501

        :param torsion_angle_constraint_violation_method: The torsion_angle_constraint_violation_method of this PdbxNmrEnsemble.  # noqa: E501
        :type: str
        """

        self._torsion_angle_constraint_violation_method = torsion_angle_constraint_violation_method

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
        if issubclass(PdbxNmrEnsemble, dict):
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
        if not isinstance(other, PdbxNmrEnsemble):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
