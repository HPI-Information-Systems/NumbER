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

class RcsbPrimaryCitation(object):
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
        'book_id_isbn': 'str',
        'book_publisher': 'str',
        'book_publisher_city': 'str',
        'book_title': 'str',
        'coordinate_linkage': 'str',
        'country': 'str',
        'id': 'str',
        'journal_abbrev': 'str',
        'journal_id_astm': 'str',
        'journal_id_csd': 'str',
        'journal_id_issn': 'str',
        'journal_issue': 'str',
        'journal_volume': 'str',
        'language': 'str',
        'page_first': 'str',
        'page_last': 'str',
        'pdbx_database_id_doi': 'str',
        'pdbx_database_id_pub_med': 'int',
        'rcsb_orcid_identifiers': 'list[str]',
        'rcsb_authors': 'list[str]',
        'rcsb_journal_abbrev': 'str',
        'title': 'str',
        'year': 'int'
    }

    attribute_map = {
        'book_id_isbn': 'book_id_ISBN',
        'book_publisher': 'book_publisher',
        'book_publisher_city': 'book_publisher_city',
        'book_title': 'book_title',
        'coordinate_linkage': 'coordinate_linkage',
        'country': 'country',
        'id': 'id',
        'journal_abbrev': 'journal_abbrev',
        'journal_id_astm': 'journal_id_ASTM',
        'journal_id_csd': 'journal_id_CSD',
        'journal_id_issn': 'journal_id_ISSN',
        'journal_issue': 'journal_issue',
        'journal_volume': 'journal_volume',
        'language': 'language',
        'page_first': 'page_first',
        'page_last': 'page_last',
        'pdbx_database_id_doi': 'pdbx_database_id_DOI',
        'pdbx_database_id_pub_med': 'pdbx_database_id_PubMed',
        'rcsb_orcid_identifiers': 'rcsb_ORCID_identifiers',
        'rcsb_authors': 'rcsb_authors',
        'rcsb_journal_abbrev': 'rcsb_journal_abbrev',
        'title': 'title',
        'year': 'year'
    }

    def __init__(self, book_id_isbn=None, book_publisher=None, book_publisher_city=None, book_title=None, coordinate_linkage=None, country=None, id=None, journal_abbrev=None, journal_id_astm=None, journal_id_csd=None, journal_id_issn=None, journal_issue=None, journal_volume=None, language=None, page_first=None, page_last=None, pdbx_database_id_doi=None, pdbx_database_id_pub_med=None, rcsb_orcid_identifiers=None, rcsb_authors=None, rcsb_journal_abbrev=None, title=None, year=None):  # noqa: E501
        """RcsbPrimaryCitation - a model defined in Swagger"""  # noqa: E501
        self._book_id_isbn = None
        self._book_publisher = None
        self._book_publisher_city = None
        self._book_title = None
        self._coordinate_linkage = None
        self._country = None
        self._id = None
        self._journal_abbrev = None
        self._journal_id_astm = None
        self._journal_id_csd = None
        self._journal_id_issn = None
        self._journal_issue = None
        self._journal_volume = None
        self._language = None
        self._page_first = None
        self._page_last = None
        self._pdbx_database_id_doi = None
        self._pdbx_database_id_pub_med = None
        self._rcsb_orcid_identifiers = None
        self._rcsb_authors = None
        self._rcsb_journal_abbrev = None
        self._title = None
        self._year = None
        self.discriminator = None
        if book_id_isbn is not None:
            self.book_id_isbn = book_id_isbn
        if book_publisher is not None:
            self.book_publisher = book_publisher
        if book_publisher_city is not None:
            self.book_publisher_city = book_publisher_city
        if book_title is not None:
            self.book_title = book_title
        if coordinate_linkage is not None:
            self.coordinate_linkage = coordinate_linkage
        if country is not None:
            self.country = country
        self.id = id
        if journal_abbrev is not None:
            self.journal_abbrev = journal_abbrev
        if journal_id_astm is not None:
            self.journal_id_astm = journal_id_astm
        if journal_id_csd is not None:
            self.journal_id_csd = journal_id_csd
        if journal_id_issn is not None:
            self.journal_id_issn = journal_id_issn
        if journal_issue is not None:
            self.journal_issue = journal_issue
        if journal_volume is not None:
            self.journal_volume = journal_volume
        if language is not None:
            self.language = language
        if page_first is not None:
            self.page_first = page_first
        if page_last is not None:
            self.page_last = page_last
        if pdbx_database_id_doi is not None:
            self.pdbx_database_id_doi = pdbx_database_id_doi
        if pdbx_database_id_pub_med is not None:
            self.pdbx_database_id_pub_med = pdbx_database_id_pub_med
        if rcsb_orcid_identifiers is not None:
            self.rcsb_orcid_identifiers = rcsb_orcid_identifiers
        if rcsb_authors is not None:
            self.rcsb_authors = rcsb_authors
        if rcsb_journal_abbrev is not None:
            self.rcsb_journal_abbrev = rcsb_journal_abbrev
        if title is not None:
            self.title = title
        if year is not None:
            self.year = year

    @property
    def book_id_isbn(self):
        """Gets the book_id_isbn of this RcsbPrimaryCitation.  # noqa: E501

        The International Standard Book Number (ISBN) code assigned to  the book cited; relevant for books or book chapters.  # noqa: E501

        :return: The book_id_isbn of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._book_id_isbn

    @book_id_isbn.setter
    def book_id_isbn(self, book_id_isbn):
        """Sets the book_id_isbn of this RcsbPrimaryCitation.

        The International Standard Book Number (ISBN) code assigned to  the book cited; relevant for books or book chapters.  # noqa: E501

        :param book_id_isbn: The book_id_isbn of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._book_id_isbn = book_id_isbn

    @property
    def book_publisher(self):
        """Gets the book_publisher of this RcsbPrimaryCitation.  # noqa: E501

        The name of the publisher of the citation; relevant  for books or book chapters.  # noqa: E501

        :return: The book_publisher of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._book_publisher

    @book_publisher.setter
    def book_publisher(self, book_publisher):
        """Sets the book_publisher of this RcsbPrimaryCitation.

        The name of the publisher of the citation; relevant  for books or book chapters.  # noqa: E501

        :param book_publisher: The book_publisher of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._book_publisher = book_publisher

    @property
    def book_publisher_city(self):
        """Gets the book_publisher_city of this RcsbPrimaryCitation.  # noqa: E501

        The location of the publisher of the citation; relevant  for books or book chapters.  # noqa: E501

        :return: The book_publisher_city of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._book_publisher_city

    @book_publisher_city.setter
    def book_publisher_city(self, book_publisher_city):
        """Sets the book_publisher_city of this RcsbPrimaryCitation.

        The location of the publisher of the citation; relevant  for books or book chapters.  # noqa: E501

        :param book_publisher_city: The book_publisher_city of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._book_publisher_city = book_publisher_city

    @property
    def book_title(self):
        """Gets the book_title of this RcsbPrimaryCitation.  # noqa: E501

        The title of the book in which the citation appeared; relevant  for books or book chapters.  # noqa: E501

        :return: The book_title of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._book_title

    @book_title.setter
    def book_title(self, book_title):
        """Sets the book_title of this RcsbPrimaryCitation.

        The title of the book in which the citation appeared; relevant  for books or book chapters.  # noqa: E501

        :param book_title: The book_title of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._book_title = book_title

    @property
    def coordinate_linkage(self):
        """Gets the coordinate_linkage of this RcsbPrimaryCitation.  # noqa: E501

        _rcsb_primary_citation.coordinate_linkage states whether this citation  is concerned with precisely the set of coordinates given in the  data block. If, for instance, the publication described the same  structure, but the coordinates had undergone further refinement  prior to the creation of the data block, the value of this data  item would be 'no'.  # noqa: E501

        :return: The coordinate_linkage of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._coordinate_linkage

    @coordinate_linkage.setter
    def coordinate_linkage(self, coordinate_linkage):
        """Sets the coordinate_linkage of this RcsbPrimaryCitation.

        _rcsb_primary_citation.coordinate_linkage states whether this citation  is concerned with precisely the set of coordinates given in the  data block. If, for instance, the publication described the same  structure, but the coordinates had undergone further refinement  prior to the creation of the data block, the value of this data  item would be 'no'.  # noqa: E501

        :param coordinate_linkage: The coordinate_linkage of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """
        allowed_values = ["n", "no", "y", "yes"]  # noqa: E501
        if coordinate_linkage not in allowed_values:
            raise ValueError(
                "Invalid value for `coordinate_linkage` ({0}), must be one of {1}"  # noqa: E501
                .format(coordinate_linkage, allowed_values)
            )

        self._coordinate_linkage = coordinate_linkage

    @property
    def country(self):
        """Gets the country of this RcsbPrimaryCitation.  # noqa: E501

        The country/region of publication; relevant for books  and book chapters.  # noqa: E501

        :return: The country of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._country

    @country.setter
    def country(self, country):
        """Sets the country of this RcsbPrimaryCitation.

        The country/region of publication; relevant for books  and book chapters.  # noqa: E501

        :param country: The country of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._country = country

    @property
    def id(self):
        """Gets the id of this RcsbPrimaryCitation.  # noqa: E501

        The value of _rcsb_primary_citation.id must uniquely identify a record in the  CITATION list.   The _rcsb_primary_citation.id 'primary' should be used to indicate the  citation that the author(s) consider to be the most pertinent to  the contents of the data block.   Note that this item need not be a number; it can be any unique  identifier.  # noqa: E501

        :return: The id of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this RcsbPrimaryCitation.

        The value of _rcsb_primary_citation.id must uniquely identify a record in the  CITATION list.   The _rcsb_primary_citation.id 'primary' should be used to indicate the  citation that the author(s) consider to be the most pertinent to  the contents of the data block.   Note that this item need not be a number; it can be any unique  identifier.  # noqa: E501

        :param id: The id of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def journal_abbrev(self):
        """Gets the journal_abbrev of this RcsbPrimaryCitation.  # noqa: E501

        Abbreviated name of the cited journal as given in the  Chemical Abstracts Service Source Index.  # noqa: E501

        :return: The journal_abbrev of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._journal_abbrev

    @journal_abbrev.setter
    def journal_abbrev(self, journal_abbrev):
        """Sets the journal_abbrev of this RcsbPrimaryCitation.

        Abbreviated name of the cited journal as given in the  Chemical Abstracts Service Source Index.  # noqa: E501

        :param journal_abbrev: The journal_abbrev of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._journal_abbrev = journal_abbrev

    @property
    def journal_id_astm(self):
        """Gets the journal_id_astm of this RcsbPrimaryCitation.  # noqa: E501

        The American Society for Testing and Materials (ASTM) code  assigned to the journal cited (also referred to as the CODEN  designator of the Chemical Abstracts Service); relevant for  journal articles.  # noqa: E501

        :return: The journal_id_astm of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._journal_id_astm

    @journal_id_astm.setter
    def journal_id_astm(self, journal_id_astm):
        """Sets the journal_id_astm of this RcsbPrimaryCitation.

        The American Society for Testing and Materials (ASTM) code  assigned to the journal cited (also referred to as the CODEN  designator of the Chemical Abstracts Service); relevant for  journal articles.  # noqa: E501

        :param journal_id_astm: The journal_id_astm of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._journal_id_astm = journal_id_astm

    @property
    def journal_id_csd(self):
        """Gets the journal_id_csd of this RcsbPrimaryCitation.  # noqa: E501

        The Cambridge Structural Database (CSD) code assigned to the  journal cited; relevant for journal articles. This is also the  system used at the Protein Data Bank (PDB).  # noqa: E501

        :return: The journal_id_csd of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._journal_id_csd

    @journal_id_csd.setter
    def journal_id_csd(self, journal_id_csd):
        """Sets the journal_id_csd of this RcsbPrimaryCitation.

        The Cambridge Structural Database (CSD) code assigned to the  journal cited; relevant for journal articles. This is also the  system used at the Protein Data Bank (PDB).  # noqa: E501

        :param journal_id_csd: The journal_id_csd of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._journal_id_csd = journal_id_csd

    @property
    def journal_id_issn(self):
        """Gets the journal_id_issn of this RcsbPrimaryCitation.  # noqa: E501

        The International Standard Serial Number (ISSN) code assigned to  the journal cited; relevant for journal articles.  # noqa: E501

        :return: The journal_id_issn of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._journal_id_issn

    @journal_id_issn.setter
    def journal_id_issn(self, journal_id_issn):
        """Sets the journal_id_issn of this RcsbPrimaryCitation.

        The International Standard Serial Number (ISSN) code assigned to  the journal cited; relevant for journal articles.  # noqa: E501

        :param journal_id_issn: The journal_id_issn of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._journal_id_issn = journal_id_issn

    @property
    def journal_issue(self):
        """Gets the journal_issue of this RcsbPrimaryCitation.  # noqa: E501

        Issue number of the journal cited; relevant for journal  articles.  # noqa: E501

        :return: The journal_issue of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._journal_issue

    @journal_issue.setter
    def journal_issue(self, journal_issue):
        """Sets the journal_issue of this RcsbPrimaryCitation.

        Issue number of the journal cited; relevant for journal  articles.  # noqa: E501

        :param journal_issue: The journal_issue of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._journal_issue = journal_issue

    @property
    def journal_volume(self):
        """Gets the journal_volume of this RcsbPrimaryCitation.  # noqa: E501

        Volume number of the journal cited; relevant for journal  articles.  # noqa: E501

        :return: The journal_volume of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._journal_volume

    @journal_volume.setter
    def journal_volume(self, journal_volume):
        """Sets the journal_volume of this RcsbPrimaryCitation.

        Volume number of the journal cited; relevant for journal  articles.  # noqa: E501

        :param journal_volume: The journal_volume of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._journal_volume = journal_volume

    @property
    def language(self):
        """Gets the language of this RcsbPrimaryCitation.  # noqa: E501

        Language in which the cited article is written.  # noqa: E501

        :return: The language of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._language

    @language.setter
    def language(self, language):
        """Sets the language of this RcsbPrimaryCitation.

        Language in which the cited article is written.  # noqa: E501

        :param language: The language of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._language = language

    @property
    def page_first(self):
        """Gets the page_first of this RcsbPrimaryCitation.  # noqa: E501

        The first page of the citation; relevant for journal  articles, books and book chapters.  # noqa: E501

        :return: The page_first of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._page_first

    @page_first.setter
    def page_first(self, page_first):
        """Sets the page_first of this RcsbPrimaryCitation.

        The first page of the citation; relevant for journal  articles, books and book chapters.  # noqa: E501

        :param page_first: The page_first of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._page_first = page_first

    @property
    def page_last(self):
        """Gets the page_last of this RcsbPrimaryCitation.  # noqa: E501

        The last page of the citation; relevant for journal  articles, books and book chapters.  # noqa: E501

        :return: The page_last of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._page_last

    @page_last.setter
    def page_last(self, page_last):
        """Sets the page_last of this RcsbPrimaryCitation.

        The last page of the citation; relevant for journal  articles, books and book chapters.  # noqa: E501

        :param page_last: The page_last of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._page_last = page_last

    @property
    def pdbx_database_id_doi(self):
        """Gets the pdbx_database_id_doi of this RcsbPrimaryCitation.  # noqa: E501

        Document Object Identifier used by doi.org to uniquely  specify bibliographic entry.  # noqa: E501

        :return: The pdbx_database_id_doi of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._pdbx_database_id_doi

    @pdbx_database_id_doi.setter
    def pdbx_database_id_doi(self, pdbx_database_id_doi):
        """Sets the pdbx_database_id_doi of this RcsbPrimaryCitation.

        Document Object Identifier used by doi.org to uniquely  specify bibliographic entry.  # noqa: E501

        :param pdbx_database_id_doi: The pdbx_database_id_doi of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._pdbx_database_id_doi = pdbx_database_id_doi

    @property
    def pdbx_database_id_pub_med(self):
        """Gets the pdbx_database_id_pub_med of this RcsbPrimaryCitation.  # noqa: E501

        Ascession number used by PubMed to categorize a specific  bibliographic entry.  # noqa: E501

        :return: The pdbx_database_id_pub_med of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_database_id_pub_med

    @pdbx_database_id_pub_med.setter
    def pdbx_database_id_pub_med(self, pdbx_database_id_pub_med):
        """Sets the pdbx_database_id_pub_med of this RcsbPrimaryCitation.

        Ascession number used by PubMed to categorize a specific  bibliographic entry.  # noqa: E501

        :param pdbx_database_id_pub_med: The pdbx_database_id_pub_med of this RcsbPrimaryCitation.  # noqa: E501
        :type: int
        """

        self._pdbx_database_id_pub_med = pdbx_database_id_pub_med

    @property
    def rcsb_orcid_identifiers(self):
        """Gets the rcsb_orcid_identifiers of this RcsbPrimaryCitation.  # noqa: E501

        The Open Researcher and Contributor ID (ORCID) identifiers for the citation authors.  # noqa: E501

        :return: The rcsb_orcid_identifiers of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: list[str]
        """
        return self._rcsb_orcid_identifiers

    @rcsb_orcid_identifiers.setter
    def rcsb_orcid_identifiers(self, rcsb_orcid_identifiers):
        """Sets the rcsb_orcid_identifiers of this RcsbPrimaryCitation.

        The Open Researcher and Contributor ID (ORCID) identifiers for the citation authors.  # noqa: E501

        :param rcsb_orcid_identifiers: The rcsb_orcid_identifiers of this RcsbPrimaryCitation.  # noqa: E501
        :type: list[str]
        """

        self._rcsb_orcid_identifiers = rcsb_orcid_identifiers

    @property
    def rcsb_authors(self):
        """Gets the rcsb_authors of this RcsbPrimaryCitation.  # noqa: E501

        Names of the authors of the citation; relevant for journal  articles, books and book chapters.  Names are separated by vertical bars.   The family name(s), followed by a comma and including any  dynastic components, precedes the first name(s) or initial(s).  # noqa: E501

        :return: The rcsb_authors of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: list[str]
        """
        return self._rcsb_authors

    @rcsb_authors.setter
    def rcsb_authors(self, rcsb_authors):
        """Sets the rcsb_authors of this RcsbPrimaryCitation.

        Names of the authors of the citation; relevant for journal  articles, books and book chapters.  Names are separated by vertical bars.   The family name(s), followed by a comma and including any  dynastic components, precedes the first name(s) or initial(s).  # noqa: E501

        :param rcsb_authors: The rcsb_authors of this RcsbPrimaryCitation.  # noqa: E501
        :type: list[str]
        """

        self._rcsb_authors = rcsb_authors

    @property
    def rcsb_journal_abbrev(self):
        """Gets the rcsb_journal_abbrev of this RcsbPrimaryCitation.  # noqa: E501

        Normalized journal abbreviation.  # noqa: E501

        :return: The rcsb_journal_abbrev of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._rcsb_journal_abbrev

    @rcsb_journal_abbrev.setter
    def rcsb_journal_abbrev(self, rcsb_journal_abbrev):
        """Sets the rcsb_journal_abbrev of this RcsbPrimaryCitation.

        Normalized journal abbreviation.  # noqa: E501

        :param rcsb_journal_abbrev: The rcsb_journal_abbrev of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._rcsb_journal_abbrev = rcsb_journal_abbrev

    @property
    def title(self):
        """Gets the title of this RcsbPrimaryCitation.  # noqa: E501

        The title of the citation; relevant for journal articles, books  and book chapters.  # noqa: E501

        :return: The title of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title):
        """Sets the title of this RcsbPrimaryCitation.

        The title of the citation; relevant for journal articles, books  and book chapters.  # noqa: E501

        :param title: The title of this RcsbPrimaryCitation.  # noqa: E501
        :type: str
        """

        self._title = title

    @property
    def year(self):
        """Gets the year of this RcsbPrimaryCitation.  # noqa: E501

        The year of the citation; relevant for journal articles, books  and book chapters.  # noqa: E501

        :return: The year of this RcsbPrimaryCitation.  # noqa: E501
        :rtype: int
        """
        return self._year

    @year.setter
    def year(self, year):
        """Sets the year of this RcsbPrimaryCitation.

        The year of the citation; relevant for journal articles, books  and book chapters.  # noqa: E501

        :param year: The year of this RcsbPrimaryCitation.  # noqa: E501
        :type: int
        """

        self._year = year

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
        if issubclass(RcsbPrimaryCitation, dict):
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
        if not isinstance(other, RcsbPrimaryCitation):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
