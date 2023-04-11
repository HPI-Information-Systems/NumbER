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

class PdbxSGProject(object):
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
        'full_name_of_center': 'str',
        'id': 'int',
        'initial_of_center': 'str',
        'project_name': 'str'
    }

    attribute_map = {
        'full_name_of_center': 'full_name_of_center',
        'id': 'id',
        'initial_of_center': 'initial_of_center',
        'project_name': 'project_name'
    }

    def __init__(self, full_name_of_center=None, id=None, initial_of_center=None, project_name=None):  # noqa: E501
        """PdbxSGProject - a model defined in Swagger"""  # noqa: E501
        self._full_name_of_center = None
        self._id = None
        self._initial_of_center = None
        self._project_name = None
        self.discriminator = None
        if full_name_of_center is not None:
            self.full_name_of_center = full_name_of_center
        self.id = id
        if initial_of_center is not None:
            self.initial_of_center = initial_of_center
        if project_name is not None:
            self.project_name = project_name

    @property
    def full_name_of_center(self):
        """Gets the full_name_of_center of this PdbxSGProject.  # noqa: E501

        The value identifies the full name of center.  # noqa: E501

        :return: The full_name_of_center of this PdbxSGProject.  # noqa: E501
        :rtype: str
        """
        return self._full_name_of_center

    @full_name_of_center.setter
    def full_name_of_center(self, full_name_of_center):
        """Sets the full_name_of_center of this PdbxSGProject.

        The value identifies the full name of center.  # noqa: E501

        :param full_name_of_center: The full_name_of_center of this PdbxSGProject.  # noqa: E501
        :type: str
        """
        allowed_values = ["Accelerated Technologies Center for Gene to 3D Structure", "Assembly, Dynamics and Evolution of Cell-Cell and Cell-Matrix Adhesions", "Atoms-to-Animals: The Immune Function Network", "Bacterial targets at IGS-CNRS, France", "Berkeley Structural Genomics Center", "Center for Eukaryotic Structural Genomics", "Center for High-Throughput Structural Biology", "Center for Membrane Proteins of Infectious Diseases", "Center for Structural Biology of Infectious Diseases", "Center for Structural Genomics of Infectious Diseases", "Center for Structures of Membrane Proteins", "Center for the X-ray Structure Determination of Human Transporters", "Chaperone-Enabled Studies of Epigenetic Regulation Enzymes", "Enzyme Discovery for Natural Product Biosynthesis", "GPCR Network", "Integrated Center for Structure and Function Innovation", "Israel Structural Proteomics Center", "Joint Center for Structural Genomics", "Marseilles Structural Genomics Program @ AFMB", "Medical Structural Genomics of Pathogenic Protozoa", "Membrane Protein Structural Biology Consortium", "Membrane Protein Structures by Solution NMR", "Midwest Center for Macromolecular Research", "Midwest Center for Structural Genomics", "Mitochondrial Protein Partnership", "Montreal-Kingston Bacterial Structural Genomics Initiative", "Mycobacterium Tuberculosis Structural Proteomics Project", "New York Consortium on Membrane Protein Structure", "New York SGX Research Center for Structural Genomics", "New York Structural GenomiX Research Consortium", "New York Structural Genomics Research Consortium", "Northeast Structural Genomics Consortium", "Nucleocytoplasmic Transport: a Target for Cellular Control", "Ontario Centre for Structural Proteomics", "Oxford Protein Production Facility", "Paris-Sud Yeast Structural Genomics", "Partnership for Nuclear Receptor Signaling Code Biology", "Partnership for Stem Cell Biology", "Partnership for T-Cell Biology", "Program for the Characterization of Secreted Effector Proteins", "Protein Structure Factory", "RIKEN Structural Genomics/Proteomics Initiative", "Scottish Structural Proteomics Facility", "Seattle Structural Genomics Center for Infectious Disease", "South Africa Structural Targets Annotation Database", "Southeast Collaboratory for Structural Genomics", "Structural Genomics Consortium", "Structural Genomics Consortium for Research on Gene Expression", "Structural Genomics of Pathogenic Protozoa Consortium", "Structural Proteomics in Europe", "Structural Proteomics in Europe 2", "Structure 2 Function Project", "Structure, Dynamics and Activation Mechanisms of Chemokine Receptors", "Structure-Function Analysis of Polymorphic CDI Toxin-Immunity Protein Complexes", "Structure-Function Studies of Tight Junction Membrane Proteins", "Structures of Mtb Proteins Conferring Susceptibility to Known Mtb Inhibitors", "TB Structural Genomics Consortium", "Transcontinental EM Initiative for Membrane Protein Structure", "Transmembrane Protein Center"]  # noqa: E501
        if full_name_of_center not in allowed_values:
            raise ValueError(
                "Invalid value for `full_name_of_center` ({0}), must be one of {1}"  # noqa: E501
                .format(full_name_of_center, allowed_values)
            )

        self._full_name_of_center = full_name_of_center

    @property
    def id(self):
        """Gets the id of this PdbxSGProject.  # noqa: E501

        A unique integer identifier for this center  # noqa: E501

        :return: The id of this PdbxSGProject.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this PdbxSGProject.

        A unique integer identifier for this center  # noqa: E501

        :param id: The id of this PdbxSGProject.  # noqa: E501
        :type: int
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501
        allowed_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # noqa: E501
        if id not in allowed_values:
            raise ValueError(
                "Invalid value for `id` ({0}), must be one of {1}"  # noqa: E501
                .format(id, allowed_values)
            )

        self._id = id

    @property
    def initial_of_center(self):
        """Gets the initial_of_center of this PdbxSGProject.  # noqa: E501

        The value identifies the full name of center.  # noqa: E501

        :return: The initial_of_center of this PdbxSGProject.  # noqa: E501
        :rtype: str
        """
        return self._initial_of_center

    @initial_of_center.setter
    def initial_of_center(self, initial_of_center):
        """Sets the initial_of_center of this PdbxSGProject.

        The value identifies the full name of center.  # noqa: E501

        :param initial_of_center: The initial_of_center of this PdbxSGProject.  # noqa: E501
        :type: str
        """
        allowed_values = ["ATCG3D", "BIGS", "BSGC", "BSGI", "CEBS", "CELLMAT", "CESG", "CHSAM", "CHTSB", "CSBID", "CSGID", "CSMP", "GPCR", "IFN", "ISFI", "ISPC", "JCSG", "MCMR", "MCSG", "MPID", "MPP", "MPSBC", "MPSbyNMR", "MSGP", "MSGPP", "MTBI", "NESG", "NHRs", "NPCXstals", "NYCOMPS", "NYSGRC", "NYSGXRC", "NatPro", "OCSP", "OPPF", "PCSEP", "PSF", "RSGI", "S2F", "SASTAD", "SECSG", "SGC", "SGCGES", "SGPP", "SPINE", "SPINE-2", "SSGCID", "SSPF", "STEMCELL", "TBSGC", "TCELL", "TEMIMPS", "TJMP", "TMPC", "TransportPDB", "UC4CDI", "XMTB", "YSG"]  # noqa: E501
        if initial_of_center not in allowed_values:
            raise ValueError(
                "Invalid value for `initial_of_center` ({0}), must be one of {1}"  # noqa: E501
                .format(initial_of_center, allowed_values)
            )

        self._initial_of_center = initial_of_center

    @property
    def project_name(self):
        """Gets the project_name of this PdbxSGProject.  # noqa: E501

        The value identifies the Structural Genomics project.  # noqa: E501

        :return: The project_name of this PdbxSGProject.  # noqa: E501
        :rtype: str
        """
        return self._project_name

    @project_name.setter
    def project_name(self, project_name):
        """Sets the project_name of this PdbxSGProject.

        The value identifies the Structural Genomics project.  # noqa: E501

        :param project_name: The project_name of this PdbxSGProject.  # noqa: E501
        :type: str
        """
        allowed_values = ["Enzyme Function Initiative", "NIAID, National Institute of Allergy and Infectious Diseases", "NPPSFA, National Project on Protein Structural and Functional Analyses", "PSI, Protein Structure Initiative", "PSI:Biology"]  # noqa: E501
        if project_name not in allowed_values:
            raise ValueError(
                "Invalid value for `project_name` ({0}), must be one of {1}"  # noqa: E501
                .format(project_name, allowed_values)
            )

        self._project_name = project_name

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
        if issubclass(PdbxSGProject, dict):
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
        if not isinstance(other, PdbxSGProject):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
