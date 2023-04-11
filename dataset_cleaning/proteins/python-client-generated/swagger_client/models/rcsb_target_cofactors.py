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

class RcsbTargetCofactors(object):
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
        'binding_assay_value': 'float',
        'binding_assay_value_type': 'str',
        'cofactor_in_ch_i_key': 'str',
        'cofactor_smiles': 'str',
        'cofactor_chem_comp_id': 'str',
        'cofactor_description': 'str',
        'cofactor_name': 'str',
        'cofactor_prd_id': 'str',
        'cofactor_resource_id': 'str',
        'mechanism_of_action': 'str',
        'neighbor_flag': 'str',
        'patent_nos': 'list[str]',
        'pubmed_ids': 'list[int]',
        'resource_name': 'str',
        'resource_version': 'str',
        'target_resource_id': 'str'
    }

    attribute_map = {
        'binding_assay_value': 'binding_assay_value',
        'binding_assay_value_type': 'binding_assay_value_type',
        'cofactor_in_ch_i_key': 'cofactor_InChIKey',
        'cofactor_smiles': 'cofactor_SMILES',
        'cofactor_chem_comp_id': 'cofactor_chem_comp_id',
        'cofactor_description': 'cofactor_description',
        'cofactor_name': 'cofactor_name',
        'cofactor_prd_id': 'cofactor_prd_id',
        'cofactor_resource_id': 'cofactor_resource_id',
        'mechanism_of_action': 'mechanism_of_action',
        'neighbor_flag': 'neighbor_flag',
        'patent_nos': 'patent_nos',
        'pubmed_ids': 'pubmed_ids',
        'resource_name': 'resource_name',
        'resource_version': 'resource_version',
        'target_resource_id': 'target_resource_id'
    }

    def __init__(self, binding_assay_value=None, binding_assay_value_type=None, cofactor_in_ch_i_key=None, cofactor_smiles=None, cofactor_chem_comp_id=None, cofactor_description=None, cofactor_name=None, cofactor_prd_id=None, cofactor_resource_id=None, mechanism_of_action=None, neighbor_flag=None, patent_nos=None, pubmed_ids=None, resource_name=None, resource_version=None, target_resource_id=None):  # noqa: E501
        """RcsbTargetCofactors - a model defined in Swagger"""  # noqa: E501
        self._binding_assay_value = None
        self._binding_assay_value_type = None
        self._cofactor_in_ch_i_key = None
        self._cofactor_smiles = None
        self._cofactor_chem_comp_id = None
        self._cofactor_description = None
        self._cofactor_name = None
        self._cofactor_prd_id = None
        self._cofactor_resource_id = None
        self._mechanism_of_action = None
        self._neighbor_flag = None
        self._patent_nos = None
        self._pubmed_ids = None
        self._resource_name = None
        self._resource_version = None
        self._target_resource_id = None
        self.discriminator = None
        if binding_assay_value is not None:
            self.binding_assay_value = binding_assay_value
        if binding_assay_value_type is not None:
            self.binding_assay_value_type = binding_assay_value_type
        if cofactor_in_ch_i_key is not None:
            self.cofactor_in_ch_i_key = cofactor_in_ch_i_key
        if cofactor_smiles is not None:
            self.cofactor_smiles = cofactor_smiles
        if cofactor_chem_comp_id is not None:
            self.cofactor_chem_comp_id = cofactor_chem_comp_id
        if cofactor_description is not None:
            self.cofactor_description = cofactor_description
        if cofactor_name is not None:
            self.cofactor_name = cofactor_name
        if cofactor_prd_id is not None:
            self.cofactor_prd_id = cofactor_prd_id
        if cofactor_resource_id is not None:
            self.cofactor_resource_id = cofactor_resource_id
        if mechanism_of_action is not None:
            self.mechanism_of_action = mechanism_of_action
        if neighbor_flag is not None:
            self.neighbor_flag = neighbor_flag
        if patent_nos is not None:
            self.patent_nos = patent_nos
        if pubmed_ids is not None:
            self.pubmed_ids = pubmed_ids
        if resource_name is not None:
            self.resource_name = resource_name
        if resource_version is not None:
            self.resource_version = resource_version
        if target_resource_id is not None:
            self.target_resource_id = target_resource_id

    @property
    def binding_assay_value(self):
        """Gets the binding_assay_value of this RcsbTargetCofactors.  # noqa: E501

        The value measured or determined by the assay.  # noqa: E501

        :return: The binding_assay_value of this RcsbTargetCofactors.  # noqa: E501
        :rtype: float
        """
        return self._binding_assay_value

    @binding_assay_value.setter
    def binding_assay_value(self, binding_assay_value):
        """Sets the binding_assay_value of this RcsbTargetCofactors.

        The value measured or determined by the assay.  # noqa: E501

        :param binding_assay_value: The binding_assay_value of this RcsbTargetCofactors.  # noqa: E501
        :type: float
        """

        self._binding_assay_value = binding_assay_value

    @property
    def binding_assay_value_type(self):
        """Gets the binding_assay_value_type of this RcsbTargetCofactors.  # noqa: E501

        The type of measurement or value determined by the assay.  # noqa: E501

        :return: The binding_assay_value_type of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._binding_assay_value_type

    @binding_assay_value_type.setter
    def binding_assay_value_type(self, binding_assay_value_type):
        """Sets the binding_assay_value_type of this RcsbTargetCofactors.

        The type of measurement or value determined by the assay.  # noqa: E501

        :param binding_assay_value_type: The binding_assay_value_type of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """
        allowed_values = ["pAC50", "pEC50", "pIC50", "pKd", "pKi"]  # noqa: E501
        if binding_assay_value_type not in allowed_values:
            raise ValueError(
                "Invalid value for `binding_assay_value_type` ({0}), must be one of {1}"  # noqa: E501
                .format(binding_assay_value_type, allowed_values)
            )

        self._binding_assay_value_type = binding_assay_value_type

    @property
    def cofactor_in_ch_i_key(self):
        """Gets the cofactor_in_ch_i_key of this RcsbTargetCofactors.  # noqa: E501

        Standard IUPAC International Chemical Identifier (InChI) descriptor key  for the cofactor.   InChI, the IUPAC International Chemical Identifier,  by Stephen R Heller, Alan McNaught, Igor Pletnev, Stephen Stein and Dmitrii Tchekhovskoi,  Journal of Cheminformatics, 2015, 7:23  # noqa: E501

        :return: The cofactor_in_ch_i_key of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._cofactor_in_ch_i_key

    @cofactor_in_ch_i_key.setter
    def cofactor_in_ch_i_key(self, cofactor_in_ch_i_key):
        """Sets the cofactor_in_ch_i_key of this RcsbTargetCofactors.

        Standard IUPAC International Chemical Identifier (InChI) descriptor key  for the cofactor.   InChI, the IUPAC International Chemical Identifier,  by Stephen R Heller, Alan McNaught, Igor Pletnev, Stephen Stein and Dmitrii Tchekhovskoi,  Journal of Cheminformatics, 2015, 7:23  # noqa: E501

        :param cofactor_in_ch_i_key: The cofactor_in_ch_i_key of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._cofactor_in_ch_i_key = cofactor_in_ch_i_key

    @property
    def cofactor_smiles(self):
        """Gets the cofactor_smiles of this RcsbTargetCofactors.  # noqa: E501

        Simplified molecular-input line-entry system (SMILES) descriptor for the cofactor.     Weininger D (February 1988). \"SMILES, a chemical language and information system. 1.    Introduction to methodology and encoding rules\". Journal of Chemical Information and Modeling. 28 (1): 31-6.     Weininger D, Weininger A, Weininger JL (May 1989).    \"SMILES. 2. Algorithm for generation of unique SMILES notation\",    Journal of Chemical Information and Modeling. 29 (2): 97-101.  # noqa: E501

        :return: The cofactor_smiles of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._cofactor_smiles

    @cofactor_smiles.setter
    def cofactor_smiles(self, cofactor_smiles):
        """Sets the cofactor_smiles of this RcsbTargetCofactors.

        Simplified molecular-input line-entry system (SMILES) descriptor for the cofactor.     Weininger D (February 1988). \"SMILES, a chemical language and information system. 1.    Introduction to methodology and encoding rules\". Journal of Chemical Information and Modeling. 28 (1): 31-6.     Weininger D, Weininger A, Weininger JL (May 1989).    \"SMILES. 2. Algorithm for generation of unique SMILES notation\",    Journal of Chemical Information and Modeling. 29 (2): 97-101.  # noqa: E501

        :param cofactor_smiles: The cofactor_smiles of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._cofactor_smiles = cofactor_smiles

    @property
    def cofactor_chem_comp_id(self):
        """Gets the cofactor_chem_comp_id of this RcsbTargetCofactors.  # noqa: E501

        The chemical component definition identifier for the cofactor.  # noqa: E501

        :return: The cofactor_chem_comp_id of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._cofactor_chem_comp_id

    @cofactor_chem_comp_id.setter
    def cofactor_chem_comp_id(self, cofactor_chem_comp_id):
        """Sets the cofactor_chem_comp_id of this RcsbTargetCofactors.

        The chemical component definition identifier for the cofactor.  # noqa: E501

        :param cofactor_chem_comp_id: The cofactor_chem_comp_id of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._cofactor_chem_comp_id = cofactor_chem_comp_id

    @property
    def cofactor_description(self):
        """Gets the cofactor_description of this RcsbTargetCofactors.  # noqa: E501

        The cofactor description.  # noqa: E501

        :return: The cofactor_description of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._cofactor_description

    @cofactor_description.setter
    def cofactor_description(self, cofactor_description):
        """Sets the cofactor_description of this RcsbTargetCofactors.

        The cofactor description.  # noqa: E501

        :param cofactor_description: The cofactor_description of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._cofactor_description = cofactor_description

    @property
    def cofactor_name(self):
        """Gets the cofactor_name of this RcsbTargetCofactors.  # noqa: E501

        The cofactor name.  # noqa: E501

        :return: The cofactor_name of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._cofactor_name

    @cofactor_name.setter
    def cofactor_name(self, cofactor_name):
        """Sets the cofactor_name of this RcsbTargetCofactors.

        The cofactor name.  # noqa: E501

        :param cofactor_name: The cofactor_name of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._cofactor_name = cofactor_name

    @property
    def cofactor_prd_id(self):
        """Gets the cofactor_prd_id of this RcsbTargetCofactors.  # noqa: E501

        The BIRD definition identifier for the cofactor.  # noqa: E501

        :return: The cofactor_prd_id of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._cofactor_prd_id

    @cofactor_prd_id.setter
    def cofactor_prd_id(self, cofactor_prd_id):
        """Sets the cofactor_prd_id of this RcsbTargetCofactors.

        The BIRD definition identifier for the cofactor.  # noqa: E501

        :param cofactor_prd_id: The cofactor_prd_id of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._cofactor_prd_id = cofactor_prd_id

    @property
    def cofactor_resource_id(self):
        """Gets the cofactor_resource_id of this RcsbTargetCofactors.  # noqa: E501

        Identifier for the cofactor assigned by the resource.  # noqa: E501

        :return: The cofactor_resource_id of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._cofactor_resource_id

    @cofactor_resource_id.setter
    def cofactor_resource_id(self, cofactor_resource_id):
        """Sets the cofactor_resource_id of this RcsbTargetCofactors.

        Identifier for the cofactor assigned by the resource.  # noqa: E501

        :param cofactor_resource_id: The cofactor_resource_id of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._cofactor_resource_id = cofactor_resource_id

    @property
    def mechanism_of_action(self):
        """Gets the mechanism_of_action of this RcsbTargetCofactors.  # noqa: E501

        Mechanism of action describes the biochemical interaction through which the  cofactor produces a pharmacological effect.  # noqa: E501

        :return: The mechanism_of_action of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._mechanism_of_action

    @mechanism_of_action.setter
    def mechanism_of_action(self, mechanism_of_action):
        """Sets the mechanism_of_action of this RcsbTargetCofactors.

        Mechanism of action describes the biochemical interaction through which the  cofactor produces a pharmacological effect.  # noqa: E501

        :param mechanism_of_action: The mechanism_of_action of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._mechanism_of_action = mechanism_of_action

    @property
    def neighbor_flag(self):
        """Gets the neighbor_flag of this RcsbTargetCofactors.  # noqa: E501

        A flag to indicate the cofactor is a structural neighbor of this  entity.  # noqa: E501

        :return: The neighbor_flag of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._neighbor_flag

    @neighbor_flag.setter
    def neighbor_flag(self, neighbor_flag):
        """Sets the neighbor_flag of this RcsbTargetCofactors.

        A flag to indicate the cofactor is a structural neighbor of this  entity.  # noqa: E501

        :param neighbor_flag: The neighbor_flag of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """
        allowed_values = ["N", "Y"]  # noqa: E501
        if neighbor_flag not in allowed_values:
            raise ValueError(
                "Invalid value for `neighbor_flag` ({0}), must be one of {1}"  # noqa: E501
                .format(neighbor_flag, allowed_values)
            )

        self._neighbor_flag = neighbor_flag

    @property
    def patent_nos(self):
        """Gets the patent_nos of this RcsbTargetCofactors.  # noqa: E501

        Patent numbers reporting the pharmacology or activity data.  # noqa: E501

        :return: The patent_nos of this RcsbTargetCofactors.  # noqa: E501
        :rtype: list[str]
        """
        return self._patent_nos

    @patent_nos.setter
    def patent_nos(self, patent_nos):
        """Sets the patent_nos of this RcsbTargetCofactors.

        Patent numbers reporting the pharmacology or activity data.  # noqa: E501

        :param patent_nos: The patent_nos of this RcsbTargetCofactors.  # noqa: E501
        :type: list[str]
        """

        self._patent_nos = patent_nos

    @property
    def pubmed_ids(self):
        """Gets the pubmed_ids of this RcsbTargetCofactors.  # noqa: E501

        PubMed identifiers for literature supporting the pharmacology or activity data.  # noqa: E501

        :return: The pubmed_ids of this RcsbTargetCofactors.  # noqa: E501
        :rtype: list[int]
        """
        return self._pubmed_ids

    @pubmed_ids.setter
    def pubmed_ids(self, pubmed_ids):
        """Sets the pubmed_ids of this RcsbTargetCofactors.

        PubMed identifiers for literature supporting the pharmacology or activity data.  # noqa: E501

        :param pubmed_ids: The pubmed_ids of this RcsbTargetCofactors.  # noqa: E501
        :type: list[int]
        """

        self._pubmed_ids = pubmed_ids

    @property
    def resource_name(self):
        """Gets the resource_name of this RcsbTargetCofactors.  # noqa: E501

        Resource providing target and cofactor data.  # noqa: E501

        :return: The resource_name of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._resource_name

    @resource_name.setter
    def resource_name(self, resource_name):
        """Sets the resource_name of this RcsbTargetCofactors.

        Resource providing target and cofactor data.  # noqa: E501

        :param resource_name: The resource_name of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """
        allowed_values = ["ChEMBL", "DrugBank", "Pharos"]  # noqa: E501
        if resource_name not in allowed_values:
            raise ValueError(
                "Invalid value for `resource_name` ({0}), must be one of {1}"  # noqa: E501
                .format(resource_name, allowed_values)
            )

        self._resource_name = resource_name

    @property
    def resource_version(self):
        """Gets the resource_version of this RcsbTargetCofactors.  # noqa: E501

        Version of the information distributed by the data resource.  # noqa: E501

        :return: The resource_version of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._resource_version

    @resource_version.setter
    def resource_version(self, resource_version):
        """Sets the resource_version of this RcsbTargetCofactors.

        Version of the information distributed by the data resource.  # noqa: E501

        :param resource_version: The resource_version of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._resource_version = resource_version

    @property
    def target_resource_id(self):
        """Gets the target_resource_id of this RcsbTargetCofactors.  # noqa: E501

        Identifier for the target assigned by the resource.  # noqa: E501

        :return: The target_resource_id of this RcsbTargetCofactors.  # noqa: E501
        :rtype: str
        """
        return self._target_resource_id

    @target_resource_id.setter
    def target_resource_id(self, target_resource_id):
        """Sets the target_resource_id of this RcsbTargetCofactors.

        Identifier for the target assigned by the resource.  # noqa: E501

        :param target_resource_id: The target_resource_id of this RcsbTargetCofactors.  # noqa: E501
        :type: str
        """

        self._target_resource_id = target_resource_id

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
        if issubclass(RcsbTargetCofactors, dict):
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
        if not isinstance(other, RcsbTargetCofactors):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other