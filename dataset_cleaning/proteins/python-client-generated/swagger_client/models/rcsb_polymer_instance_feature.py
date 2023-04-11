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

class RcsbPolymerInstanceFeature(object):
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
        'assignment_version': 'str',
        'description': 'str',
        'feature_id': 'str',
        'name': 'str',
        'ordinal': 'int',
        'provenance_source': 'str',
        'reference_scheme': 'str',
        'type': 'str',
        'feature_positions': 'list[RcsbPolymerInstanceFeatureFeaturePositions]',
        'additional_properties': 'list[RcsbPolymerInstanceFeatureAdditionalProperties]'
    }

    attribute_map = {
        'assignment_version': 'assignment_version',
        'description': 'description',
        'feature_id': 'feature_id',
        'name': 'name',
        'ordinal': 'ordinal',
        'provenance_source': 'provenance_source',
        'reference_scheme': 'reference_scheme',
        'type': 'type',
        'feature_positions': 'feature_positions',
        'additional_properties': 'additional_properties'
    }

    def __init__(self, assignment_version=None, description=None, feature_id=None, name=None, ordinal=None, provenance_source=None, reference_scheme=None, type=None, feature_positions=None, additional_properties=None):  # noqa: E501
        """RcsbPolymerInstanceFeature - a model defined in Swagger"""  # noqa: E501
        self._assignment_version = None
        self._description = None
        self._feature_id = None
        self._name = None
        self._ordinal = None
        self._provenance_source = None
        self._reference_scheme = None
        self._type = None
        self._feature_positions = None
        self._additional_properties = None
        self.discriminator = None
        if assignment_version is not None:
            self.assignment_version = assignment_version
        if description is not None:
            self.description = description
        if feature_id is not None:
            self.feature_id = feature_id
        if name is not None:
            self.name = name
        self.ordinal = ordinal
        if provenance_source is not None:
            self.provenance_source = provenance_source
        if reference_scheme is not None:
            self.reference_scheme = reference_scheme
        if type is not None:
            self.type = type
        if feature_positions is not None:
            self.feature_positions = feature_positions
        if additional_properties is not None:
            self.additional_properties = additional_properties

    @property
    def assignment_version(self):
        """Gets the assignment_version of this RcsbPolymerInstanceFeature.  # noqa: E501

        Identifies the version of the feature assignment.  # noqa: E501

        :return: The assignment_version of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: str
        """
        return self._assignment_version

    @assignment_version.setter
    def assignment_version(self, assignment_version):
        """Sets the assignment_version of this RcsbPolymerInstanceFeature.

        Identifies the version of the feature assignment.  # noqa: E501

        :param assignment_version: The assignment_version of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: str
        """

        self._assignment_version = assignment_version

    @property
    def description(self):
        """Gets the description of this RcsbPolymerInstanceFeature.  # noqa: E501

        A description for the feature.  # noqa: E501

        :return: The description of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this RcsbPolymerInstanceFeature.

        A description for the feature.  # noqa: E501

        :param description: The description of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def feature_id(self):
        """Gets the feature_id of this RcsbPolymerInstanceFeature.  # noqa: E501

        An identifier for the feature.  # noqa: E501

        :return: The feature_id of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: str
        """
        return self._feature_id

    @feature_id.setter
    def feature_id(self, feature_id):
        """Sets the feature_id of this RcsbPolymerInstanceFeature.

        An identifier for the feature.  # noqa: E501

        :param feature_id: The feature_id of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: str
        """

        self._feature_id = feature_id

    @property
    def name(self):
        """Gets the name of this RcsbPolymerInstanceFeature.  # noqa: E501

        A name for the feature.  # noqa: E501

        :return: The name of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this RcsbPolymerInstanceFeature.

        A name for the feature.  # noqa: E501

        :param name: The name of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def ordinal(self):
        """Gets the ordinal of this RcsbPolymerInstanceFeature.  # noqa: E501

        Ordinal identifier for this category  # noqa: E501

        :return: The ordinal of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: int
        """
        return self._ordinal

    @ordinal.setter
    def ordinal(self, ordinal):
        """Sets the ordinal of this RcsbPolymerInstanceFeature.

        Ordinal identifier for this category  # noqa: E501

        :param ordinal: The ordinal of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: int
        """
        if ordinal is None:
            raise ValueError("Invalid value for `ordinal`, must not be `None`")  # noqa: E501

        self._ordinal = ordinal

    @property
    def provenance_source(self):
        """Gets the provenance_source of this RcsbPolymerInstanceFeature.  # noqa: E501

        Code identifying the individual, organization or program that  assigned the feature.  # noqa: E501

        :return: The provenance_source of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: str
        """
        return self._provenance_source

    @provenance_source.setter
    def provenance_source(self, provenance_source):
        """Sets the provenance_source of this RcsbPolymerInstanceFeature.

        Code identifying the individual, organization or program that  assigned the feature.  # noqa: E501

        :param provenance_source: The provenance_source of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: str
        """

        self._provenance_source = provenance_source

    @property
    def reference_scheme(self):
        """Gets the reference_scheme of this RcsbPolymerInstanceFeature.  # noqa: E501

        Code residue coordinate system for the assigned feature.  # noqa: E501

        :return: The reference_scheme of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: str
        """
        return self._reference_scheme

    @reference_scheme.setter
    def reference_scheme(self, reference_scheme):
        """Sets the reference_scheme of this RcsbPolymerInstanceFeature.

        Code residue coordinate system for the assigned feature.  # noqa: E501

        :param reference_scheme: The reference_scheme of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: str
        """
        allowed_values = ["NCBI", "PDB entity", "PDB entry", "UniProt"]  # noqa: E501
        if reference_scheme not in allowed_values:
            raise ValueError(
                "Invalid value for `reference_scheme` ({0}), must be one of {1}"  # noqa: E501
                .format(reference_scheme, allowed_values)
            )

        self._reference_scheme = reference_scheme

    @property
    def type(self):
        """Gets the type of this RcsbPolymerInstanceFeature.  # noqa: E501

        A type or category of the feature.  # noqa: E501

        :return: The type of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this RcsbPolymerInstanceFeature.

        A type or category of the feature.  # noqa: E501

        :param type: The type of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: str
        """
        allowed_values = ["ANGLE_OUTLIER", "BEND", "BINDING_SITE", "BOND_OUTLIER", "C-MANNOSYLATION_SITE", "CATH", "CIS-PEPTIDE", "ECOD", "HELIX_P", "HELX_LH_PP_P", "HELX_RH_3T_P", "HELX_RH_AL_P", "HELX_RH_PI_P", "MA_QA_METRIC_LOCAL_TYPE_CONTACT_PROBABILITY", "MA_QA_METRIC_LOCAL_TYPE_DISTANCE", "MA_QA_METRIC_LOCAL_TYPE_ENERGY", "MA_QA_METRIC_LOCAL_TYPE_IPTM", "MA_QA_METRIC_LOCAL_TYPE_NORMALIZED_SCORE", "MA_QA_METRIC_LOCAL_TYPE_OTHER", "MA_QA_METRIC_LOCAL_TYPE_PAE", "MA_QA_METRIC_LOCAL_TYPE_PLDDT", "MA_QA_METRIC_LOCAL_TYPE_PLDDT_ALL-ATOM", "MA_QA_METRIC_LOCAL_TYPE_PLDDT_ALL-ATOM_[0,1]", "MA_QA_METRIC_LOCAL_TYPE_PLDDT_[0,1]", "MA_QA_METRIC_LOCAL_TYPE_PTM", "MA_QA_METRIC_LOCAL_TYPE_ZSCORE", "MEMBRANE_SEGMENT", "MOGUL_ANGLE_OUTLIER", "MOGUL_BOND_OUTLIER", "N-GLYCOSYLATION_SITE", "O-GLYCOSYLATION_SITE", "RAMACHANDRAN_OUTLIER", "ROTAMER_OUTLIER", "RSCC_OUTLIER", "RSRZ_OUTLIER", "S-GLYCOSYLATION_SITE", "SABDAB_ANTIBODY_HEAVY_CHAIN_SUBCLASS", "SABDAB_ANTIBODY_LIGHT_CHAIN_SUBCLASS", "SABDAB_ANTIBODY_LIGHT_CHAIN_TYPE", "SCOP", "SCOP2B_SUPERFAMILY", "SCOP2_FAMILY", "SCOP2_SUPERFAMILY", "SHEET", "STEREO_OUTLIER", "STRN", "TURN_TY1_P", "UNASSIGNED_SEC_STRUCT", "UNOBSERVED_ATOM_XYZ", "UNOBSERVED_RESIDUE_XYZ", "ZERO_OCCUPANCY_ATOM_XYZ", "ZERO_OCCUPANCY_RESIDUE_XYZ", "ASA"]  # noqa: E501
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` ({0}), must be one of {1}"  # noqa: E501
                .format(type, allowed_values)
            )

        self._type = type

    @property
    def feature_positions(self):
        """Gets the feature_positions of this RcsbPolymerInstanceFeature.  # noqa: E501


        :return: The feature_positions of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: list[RcsbPolymerInstanceFeatureFeaturePositions]
        """
        return self._feature_positions

    @feature_positions.setter
    def feature_positions(self, feature_positions):
        """Sets the feature_positions of this RcsbPolymerInstanceFeature.


        :param feature_positions: The feature_positions of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: list[RcsbPolymerInstanceFeatureFeaturePositions]
        """

        self._feature_positions = feature_positions

    @property
    def additional_properties(self):
        """Gets the additional_properties of this RcsbPolymerInstanceFeature.  # noqa: E501


        :return: The additional_properties of this RcsbPolymerInstanceFeature.  # noqa: E501
        :rtype: list[RcsbPolymerInstanceFeatureAdditionalProperties]
        """
        return self._additional_properties

    @additional_properties.setter
    def additional_properties(self, additional_properties):
        """Sets the additional_properties of this RcsbPolymerInstanceFeature.


        :param additional_properties: The additional_properties of this RcsbPolymerInstanceFeature.  # noqa: E501
        :type: list[RcsbPolymerInstanceFeatureAdditionalProperties]
        """

        self._additional_properties = additional_properties

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
        if issubclass(RcsbPolymerInstanceFeature, dict):
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
        if not isinstance(other, RcsbPolymerInstanceFeature):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
