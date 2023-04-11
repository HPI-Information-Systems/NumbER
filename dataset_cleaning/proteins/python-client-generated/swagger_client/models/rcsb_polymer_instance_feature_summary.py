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

class RcsbPolymerInstanceFeatureSummary(object):
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
        'count': 'int',
        'coverage': 'float',
        'maximum_length': 'int',
        'maximum_value': 'float',
        'minimum_length': 'int',
        'minimum_value': 'float',
        'type': 'str'
    }

    attribute_map = {
        'count': 'count',
        'coverage': 'coverage',
        'maximum_length': 'maximum_length',
        'maximum_value': 'maximum_value',
        'minimum_length': 'minimum_length',
        'minimum_value': 'minimum_value',
        'type': 'type'
    }

    def __init__(self, count=None, coverage=None, maximum_length=None, maximum_value=None, minimum_length=None, minimum_value=None, type=None):  # noqa: E501
        """RcsbPolymerInstanceFeatureSummary - a model defined in Swagger"""  # noqa: E501
        self._count = None
        self._coverage = None
        self._maximum_length = None
        self._maximum_value = None
        self._minimum_length = None
        self._minimum_value = None
        self._type = None
        self.discriminator = None
        if count is not None:
            self.count = count
        if coverage is not None:
            self.coverage = coverage
        if maximum_length is not None:
            self.maximum_length = maximum_length
        if maximum_value is not None:
            self.maximum_value = maximum_value
        if minimum_length is not None:
            self.minimum_length = minimum_length
        if minimum_value is not None:
            self.minimum_value = minimum_value
        if type is not None:
            self.type = type

    @property
    def count(self):
        """Gets the count of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501

        The feature count per polymer chain.  # noqa: E501

        :return: The count of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :rtype: int
        """
        return self._count

    @count.setter
    def count(self, count):
        """Sets the count of this RcsbPolymerInstanceFeatureSummary.

        The feature count per polymer chain.  # noqa: E501

        :param count: The count of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :type: int
        """

        self._count = count

    @property
    def coverage(self):
        """Gets the coverage of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501

        The fractional feature coverage relative to the full entity sequence.  For instance, the fraction of features such as CATH or SCOP domains, secondary structure elements,  unobserved residues, or geometrical outliers relative to the length of the entity sequence.  # noqa: E501

        :return: The coverage of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :rtype: float
        """
        return self._coverage

    @coverage.setter
    def coverage(self, coverage):
        """Sets the coverage of this RcsbPolymerInstanceFeatureSummary.

        The fractional feature coverage relative to the full entity sequence.  For instance, the fraction of features such as CATH or SCOP domains, secondary structure elements,  unobserved residues, or geometrical outliers relative to the length of the entity sequence.  # noqa: E501

        :param coverage: The coverage of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :type: float
        """

        self._coverage = coverage

    @property
    def maximum_length(self):
        """Gets the maximum_length of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501

        The maximum feature length.  # noqa: E501

        :return: The maximum_length of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :rtype: int
        """
        return self._maximum_length

    @maximum_length.setter
    def maximum_length(self, maximum_length):
        """Sets the maximum_length of this RcsbPolymerInstanceFeatureSummary.

        The maximum feature length.  # noqa: E501

        :param maximum_length: The maximum_length of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :type: int
        """

        self._maximum_length = maximum_length

    @property
    def maximum_value(self):
        """Gets the maximum_value of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501

        The maximum feature value.  # noqa: E501

        :return: The maximum_value of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :rtype: float
        """
        return self._maximum_value

    @maximum_value.setter
    def maximum_value(self, maximum_value):
        """Sets the maximum_value of this RcsbPolymerInstanceFeatureSummary.

        The maximum feature value.  # noqa: E501

        :param maximum_value: The maximum_value of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :type: float
        """

        self._maximum_value = maximum_value

    @property
    def minimum_length(self):
        """Gets the minimum_length of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501

        The minimum feature length.  # noqa: E501

        :return: The minimum_length of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :rtype: int
        """
        return self._minimum_length

    @minimum_length.setter
    def minimum_length(self, minimum_length):
        """Sets the minimum_length of this RcsbPolymerInstanceFeatureSummary.

        The minimum feature length.  # noqa: E501

        :param minimum_length: The minimum_length of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :type: int
        """

        self._minimum_length = minimum_length

    @property
    def minimum_value(self):
        """Gets the minimum_value of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501

        The minimum feature value.  # noqa: E501

        :return: The minimum_value of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :rtype: float
        """
        return self._minimum_value

    @minimum_value.setter
    def minimum_value(self, minimum_value):
        """Sets the minimum_value of this RcsbPolymerInstanceFeatureSummary.

        The minimum feature value.  # noqa: E501

        :param minimum_value: The minimum_value of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :type: float
        """

        self._minimum_value = minimum_value

    @property
    def type(self):
        """Gets the type of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501

        Type or category of the feature.  # noqa: E501

        :return: The type of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this RcsbPolymerInstanceFeatureSummary.

        Type or category of the feature.  # noqa: E501

        :param type: The type of this RcsbPolymerInstanceFeatureSummary.  # noqa: E501
        :type: str
        """
        allowed_values = ["ANGLE_OUTLIER", "BEND", "BINDING_SITE", "BOND_OUTLIER", "C-MANNOSYLATION_SITE", "CATH", "CIS-PEPTIDE", "ECOD", "HELIX_P", "HELX_LH_PP_P", "HELX_RH_3T_P", "HELX_RH_AL_P", "HELX_RH_PI_P", "MA_QA_METRIC_LOCAL_TYPE_CONTACT_PROBABILITY", "MA_QA_METRIC_LOCAL_TYPE_DISTANCE", "MA_QA_METRIC_LOCAL_TYPE_ENERGY", "MA_QA_METRIC_LOCAL_TYPE_IPTM", "MA_QA_METRIC_LOCAL_TYPE_NORMALIZED_SCORE", "MA_QA_METRIC_LOCAL_TYPE_OTHER", "MA_QA_METRIC_LOCAL_TYPE_PAE", "MA_QA_METRIC_LOCAL_TYPE_PLDDT", "MA_QA_METRIC_LOCAL_TYPE_PLDDT_ALL-ATOM", "MA_QA_METRIC_LOCAL_TYPE_PLDDT_ALL-ATOM_[0,1]", "MA_QA_METRIC_LOCAL_TYPE_PLDDT_[0,1]", "MA_QA_METRIC_LOCAL_TYPE_PTM", "MA_QA_METRIC_LOCAL_TYPE_ZSCORE", "MEMBRANE_SEGMENT", "MOGUL_ANGLE_OUTLIER", "MOGUL_BOND_OUTLIER", "N-GLYCOSYLATION_SITE", "O-GLYCOSYLATION_SITE", "RAMACHANDRAN_OUTLIER", "ROTAMER_OUTLIER", "RSCC_OUTLIER", "RSRZ_OUTLIER", "S-GLYCOSYLATION_SITE", "SABDAB_ANTIBODY_HEAVY_CHAIN_SUBCLASS", "SABDAB_ANTIBODY_LIGHT_CHAIN_SUBCLASS", "SABDAB_ANTIBODY_LIGHT_CHAIN_TYPE", "SCOP", "SCOP2B_SUPERFAMILY", "SCOP2_FAMILY", "SCOP2_SUPERFAMILY", "SHEET", "STEREO_OUTLIER", "STRN", "TURN_TY1_P", "UNASSIGNED_SEC_STRUCT", "UNOBSERVED_ATOM_XYZ", "UNOBSERVED_RESIDUE_XYZ", "ZERO_OCCUPANCY_ATOM_XYZ", "ZERO_OCCUPANCY_RESIDUE_XYZ"]  # noqa: E501
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
        if issubclass(RcsbPolymerInstanceFeatureSummary, dict):
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
        if not isinstance(other, RcsbPolymerInstanceFeatureSummary):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
