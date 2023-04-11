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

class PdbxDatabaseStatus(object):
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
        'sg_entry': 'str',
        'deposit_site': 'str',
        'methods_development_category': 'str',
        'pdb_format_compatible': 'str',
        'process_site': 'str',
        'recvd_initial_deposition_date': 'datetime',
        'status_code': 'str',
        'status_code_cs': 'str',
        'status_code_mr': 'str',
        'status_code_sf': 'str'
    }

    attribute_map = {
        'sg_entry': 'SG_entry',
        'deposit_site': 'deposit_site',
        'methods_development_category': 'methods_development_category',
        'pdb_format_compatible': 'pdb_format_compatible',
        'process_site': 'process_site',
        'recvd_initial_deposition_date': 'recvd_initial_deposition_date',
        'status_code': 'status_code',
        'status_code_cs': 'status_code_cs',
        'status_code_mr': 'status_code_mr',
        'status_code_sf': 'status_code_sf'
    }

    def __init__(self, sg_entry=None, deposit_site=None, methods_development_category=None, pdb_format_compatible=None, process_site=None, recvd_initial_deposition_date=None, status_code=None, status_code_cs=None, status_code_mr=None, status_code_sf=None):  # noqa: E501
        """PdbxDatabaseStatus - a model defined in Swagger"""  # noqa: E501
        self._sg_entry = None
        self._deposit_site = None
        self._methods_development_category = None
        self._pdb_format_compatible = None
        self._process_site = None
        self._recvd_initial_deposition_date = None
        self._status_code = None
        self._status_code_cs = None
        self._status_code_mr = None
        self._status_code_sf = None
        self.discriminator = None
        if sg_entry is not None:
            self.sg_entry = sg_entry
        if deposit_site is not None:
            self.deposit_site = deposit_site
        if methods_development_category is not None:
            self.methods_development_category = methods_development_category
        if pdb_format_compatible is not None:
            self.pdb_format_compatible = pdb_format_compatible
        if process_site is not None:
            self.process_site = process_site
        if recvd_initial_deposition_date is not None:
            self.recvd_initial_deposition_date = recvd_initial_deposition_date
        if status_code is not None:
            self.status_code = status_code
        if status_code_cs is not None:
            self.status_code_cs = status_code_cs
        if status_code_mr is not None:
            self.status_code_mr = status_code_mr
        if status_code_sf is not None:
            self.status_code_sf = status_code_sf

    @property
    def sg_entry(self):
        """Gets the sg_entry of this PdbxDatabaseStatus.  # noqa: E501

        This code indicates whether the entry belongs to  Structural Genomics Project.  # noqa: E501

        :return: The sg_entry of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._sg_entry

    @sg_entry.setter
    def sg_entry(self, sg_entry):
        """Sets the sg_entry of this PdbxDatabaseStatus.

        This code indicates whether the entry belongs to  Structural Genomics Project.  # noqa: E501

        :param sg_entry: The sg_entry of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["N", "Y"]  # noqa: E501
        if sg_entry not in allowed_values:
            raise ValueError(
                "Invalid value for `sg_entry` ({0}), must be one of {1}"  # noqa: E501
                .format(sg_entry, allowed_values)
            )

        self._sg_entry = sg_entry

    @property
    def deposit_site(self):
        """Gets the deposit_site of this PdbxDatabaseStatus.  # noqa: E501

        The site where the file was deposited.  # noqa: E501

        :return: The deposit_site of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._deposit_site

    @deposit_site.setter
    def deposit_site(self, deposit_site):
        """Sets the deposit_site of this PdbxDatabaseStatus.

        The site where the file was deposited.  # noqa: E501

        :param deposit_site: The deposit_site of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["BMRB", "BNL", "NDB", "PDBC", "PDBE", "PDBJ", "RCSB"]  # noqa: E501
        if deposit_site not in allowed_values:
            raise ValueError(
                "Invalid value for `deposit_site` ({0}), must be one of {1}"  # noqa: E501
                .format(deposit_site, allowed_values)
            )

        self._deposit_site = deposit_site

    @property
    def methods_development_category(self):
        """Gets the methods_development_category of this PdbxDatabaseStatus.  # noqa: E501

        The methods development category in which this  entry has been placed.  # noqa: E501

        :return: The methods_development_category of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._methods_development_category

    @methods_development_category.setter
    def methods_development_category(self, methods_development_category):
        """Sets the methods_development_category of this PdbxDatabaseStatus.

        The methods development category in which this  entry has been placed.  # noqa: E501

        :param methods_development_category: The methods_development_category of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["CAPRI", "CASD-NMR", "CASP", "D3R", "FoldIt", "GPCR Dock", "RNA-Puzzles"]  # noqa: E501
        if methods_development_category not in allowed_values:
            raise ValueError(
                "Invalid value for `methods_development_category` ({0}), must be one of {1}"  # noqa: E501
                .format(methods_development_category, allowed_values)
            )

        self._methods_development_category = methods_development_category

    @property
    def pdb_format_compatible(self):
        """Gets the pdb_format_compatible of this PdbxDatabaseStatus.  # noqa: E501

        A flag indicating that the entry is compatible with the PDB format.   A value of 'N' indicates that the no PDB format data file is  corresponding to this entry is available in the PDB archive.  # noqa: E501

        :return: The pdb_format_compatible of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._pdb_format_compatible

    @pdb_format_compatible.setter
    def pdb_format_compatible(self, pdb_format_compatible):
        """Sets the pdb_format_compatible of this PdbxDatabaseStatus.

        A flag indicating that the entry is compatible with the PDB format.   A value of 'N' indicates that the no PDB format data file is  corresponding to this entry is available in the PDB archive.  # noqa: E501

        :param pdb_format_compatible: The pdb_format_compatible of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["N", "Y"]  # noqa: E501
        if pdb_format_compatible not in allowed_values:
            raise ValueError(
                "Invalid value for `pdb_format_compatible` ({0}), must be one of {1}"  # noqa: E501
                .format(pdb_format_compatible, allowed_values)
            )

        self._pdb_format_compatible = pdb_format_compatible

    @property
    def process_site(self):
        """Gets the process_site of this PdbxDatabaseStatus.  # noqa: E501

        The site where the file was deposited.  # noqa: E501

        :return: The process_site of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._process_site

    @process_site.setter
    def process_site(self, process_site):
        """Sets the process_site of this PdbxDatabaseStatus.

        The site where the file was deposited.  # noqa: E501

        :param process_site: The process_site of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["BNL", "NDB", "PDBC", "PDBE", "PDBJ", "RCSB"]  # noqa: E501
        if process_site not in allowed_values:
            raise ValueError(
                "Invalid value for `process_site` ({0}), must be one of {1}"  # noqa: E501
                .format(process_site, allowed_values)
            )

        self._process_site = process_site

    @property
    def recvd_initial_deposition_date(self):
        """Gets the recvd_initial_deposition_date of this PdbxDatabaseStatus.  # noqa: E501

        The date of initial deposition.  (The first message for  deposition has been received.)  # noqa: E501

        :return: The recvd_initial_deposition_date of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: datetime
        """
        return self._recvd_initial_deposition_date

    @recvd_initial_deposition_date.setter
    def recvd_initial_deposition_date(self, recvd_initial_deposition_date):
        """Sets the recvd_initial_deposition_date of this PdbxDatabaseStatus.

        The date of initial deposition.  (The first message for  deposition has been received.)  # noqa: E501

        :param recvd_initial_deposition_date: The recvd_initial_deposition_date of this PdbxDatabaseStatus.  # noqa: E501
        :type: datetime
        """

        self._recvd_initial_deposition_date = recvd_initial_deposition_date

    @property
    def status_code(self):
        """Gets the status_code of this PdbxDatabaseStatus.  # noqa: E501

        Code for status of file.  # noqa: E501

        :return: The status_code of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._status_code

    @status_code.setter
    def status_code(self, status_code):
        """Sets the status_code of this PdbxDatabaseStatus.

        Code for status of file.  # noqa: E501

        :param status_code: The status_code of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["AUCO", "AUTH", "BIB", "DEL", "HOLD", "HPUB", "OBS", "POLC", "PROC", "REFI", "REL", "REPL", "REV", "RMVD", "TRSF", "UPD", "WAIT", "WDRN"]  # noqa: E501
        if status_code not in allowed_values:
            raise ValueError(
                "Invalid value for `status_code` ({0}), must be one of {1}"  # noqa: E501
                .format(status_code, allowed_values)
            )

        self._status_code = status_code

    @property
    def status_code_cs(self):
        """Gets the status_code_cs of this PdbxDatabaseStatus.  # noqa: E501

        Code for status of chemical shift data file.  # noqa: E501

        :return: The status_code_cs of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._status_code_cs

    @status_code_cs.setter
    def status_code_cs(self, status_code_cs):
        """Sets the status_code_cs of this PdbxDatabaseStatus.

        Code for status of chemical shift data file.  # noqa: E501

        :param status_code_cs: The status_code_cs of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["AUCO", "AUTH", "HOLD", "HPUB", "OBS", "POLC", "PROC", "REL", "REPL", "RMVD", "WAIT", "WDRN"]  # noqa: E501
        if status_code_cs not in allowed_values:
            raise ValueError(
                "Invalid value for `status_code_cs` ({0}), must be one of {1}"  # noqa: E501
                .format(status_code_cs, allowed_values)
            )

        self._status_code_cs = status_code_cs

    @property
    def status_code_mr(self):
        """Gets the status_code_mr of this PdbxDatabaseStatus.  # noqa: E501

        Code for status of NMR constraints file.  # noqa: E501

        :return: The status_code_mr of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._status_code_mr

    @status_code_mr.setter
    def status_code_mr(self, status_code_mr):
        """Sets the status_code_mr of this PdbxDatabaseStatus.

        Code for status of NMR constraints file.  # noqa: E501

        :param status_code_mr: The status_code_mr of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["AUCO", "AUTH", "HOLD", "HPUB", "OBS", "POLC", "PROC", "REL", "REPL", "RMVD", "WAIT", "WDRN"]  # noqa: E501
        if status_code_mr not in allowed_values:
            raise ValueError(
                "Invalid value for `status_code_mr` ({0}), must be one of {1}"  # noqa: E501
                .format(status_code_mr, allowed_values)
            )

        self._status_code_mr = status_code_mr

    @property
    def status_code_sf(self):
        """Gets the status_code_sf of this PdbxDatabaseStatus.  # noqa: E501

        Code for status of structure factor file.  # noqa: E501

        :return: The status_code_sf of this PdbxDatabaseStatus.  # noqa: E501
        :rtype: str
        """
        return self._status_code_sf

    @status_code_sf.setter
    def status_code_sf(self, status_code_sf):
        """Sets the status_code_sf of this PdbxDatabaseStatus.

        Code for status of structure factor file.  # noqa: E501

        :param status_code_sf: The status_code_sf of this PdbxDatabaseStatus.  # noqa: E501
        :type: str
        """
        allowed_values = ["AUTH", "HOLD", "HPUB", "OBS", "POLC", "PROC", "REL", "REPL", "RMVD", "WAIT", "WDRN"]  # noqa: E501
        if status_code_sf not in allowed_values:
            raise ValueError(
                "Invalid value for `status_code_sf` ({0}), must be one of {1}"  # noqa: E501
                .format(status_code_sf, allowed_values)
            )

        self._status_code_sf = status_code_sf

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
        if issubclass(PdbxDatabaseStatus, dict):
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
        if not isinstance(other, PdbxDatabaseStatus):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
