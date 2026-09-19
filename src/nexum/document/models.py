from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, TypedDict, Tuple, Mapping, Sequence

import numpy as np
from PIL import Image
from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, Field


class ReaderMode(str, Enum):
    BATCH = "batch"
    STREAMING = "stream"


class Table(BaseModel):
    columns: list[Any]
    rows: list[list[Any]]
    bbox: tuple[int, int, int, int] | None = None
    page: int | None = None


class Document(BaseModel):
    metadata: dict
    content: str
    images: List[Any] | dict[str, Any] = []
    tables: List[Table] | dict[str, Any] = []
    blocks: List[Any] | dict[str, Any] = []
    flowcharts: List[Any] | dict[str, Any] = []
    layout: List[Any] = []
    source: str | None = None
    pages: List[str] = []
    raw_bytes: bytes | None = None

    def to_langchain_document(self) -> LCDocument:
        meta = {
            "metadata": self.metadata,
            "images": self.images,
            "tables": self.tables,
            "blocks": self.blocks,
            "flowcharts": self.flowcharts,
            "layout": self.layout,
            "source": self.source,
            "pages": self.pages,
            "raw_bytes": self.raw_bytes,
        }

        return LCDocument(page_content=self.content, metadata=meta)


class NexumConfig(BaseModel): ...


class CSVReaderConfig(NexumConfig):
    delimiter: str = ","
    skip_rows: int = 0
    has_header: bool = True
    encoding: str | None = "utf-8"
    infer_types: bool = True


class ImageConfig(NexumConfig):
    lang: str = "eng"
    enable_binarization: bool = True
    enable_deskew: bool = True
    enable_clahe: bool = True
    enable_denoise: bool = True
    enable_sharpen: bool = True
    enable_osd: bool = True
    clahe_clip_limit: float = 40.0
    clahe_tile_size: tuple[int, int] = (8, 8)
    strip_empty_lines: bool = True
    oem: int = 1
    psm: int = 3
    binarization_method: str = "otsu"
    denoise_method: str = "bilateral"  # "bilateral", "median", "none"
    binarization_min_std: float = 10.0
    sharpen_sigma: float = 1.0
    sharpen_strength: float = 1.2
    sharpen_blur_weight: float = -0.2
    adaptive_block_size: int = 31
    adaptive_c: int = 2
    osd_min_confidence: float = 5.0
    max_pixels: int | None = 20_000_000


class ImageOCRConfig(ImageConfig):
    enable_metadata: bool = True
    return_structured: bool = False
    table_as_df: bool = True
    enable_caption: bool = True
    enable_flowchart: bool = True


class PDFOCRConfig(ImageConfig):
    min_dpi: int = 72
    upscale_factor: float = 1.5
    deskew_max_angle: float = 5.0


class ImageReaderConfig(NexumConfig):
    read_mode: ReaderMode = ReaderMode.BATCH
    ocr_config: ImageOCRConfig = Field(default_factory=ImageOCRConfig)


class PDFReaderConfig(NexumConfig):
    read_mode: ReaderMode = ReaderMode.STREAMING
    ocr_config: PDFOCRConfig = Field(default_factory=PDFOCRConfig)
    lang: str = "eng"
    max_workers: int = 4
    dpi: int = 300
    enable_ocr: bool = True
    enable_metadata: bool = True
    enable_images: bool = True
    enable_tables: bool = True


class VisualRegion(TypedDict):
    page: int
    bbox: Tuple[int, int, int, int]


class PageAnalysisResult(TypedDict):
    layout: Any
    tables: List["Table"]
    blocks: List[VisualRegion]
    images: List[VisualRegion]
    flowcharts: List[VisualRegion]


class Parsed(BaseModel):
    metadata: dict[str, Any]
    tables: list[Any] = []
    blocks: list[Any] = []
    rows: list[list[Any]] = []
    flowcharts: list[Any] = []
    caption: str | None = None


@dataclass
class ImageContext:
    pil: Image.Image
    gray: np.ndarray
    text_canvas: np.ndarray
    width: int
    height: int
    total_pixels: int
    lang: str
    psm: int


@dataclass(frozen=True, slots=True)
class CaptionInput:
    image: Image.Image
    prompt: str | None = None


@dataclass(frozen=True, slots=True)
class CaptionConfig:
    model_name: str = "Salesforce/blip-image-captioning-base"
    device: str | None = None
    max_new_tokens: int = 50


@dataclass(frozen=True, slots=True)
class ColumnSemanticResult:
    semantic_type: str
    is_pii: bool
    is_sensitive: bool


@dataclass(frozen=True, slots=True)
class TableSemanticInput:
    columns: Sequence[str]
    data: Sequence[Sequence[object]]
    sample_size: int = 15
    max_values_per_col: int = 8


@dataclass(frozen=True, slots=True)
class TableSemanticConfig:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: str | None = None
    sensitive_types: frozenset[str] = frozenset(
        {
            "religious_belief",
            "political_opinion",
            "trade_union",
            "ethnicity_or_race",
            "health_or_medical",
            "sexual_orientation",
            "genetic_or_biometric",
        }
    )
    pii_types: frozenset[str] = frozenset(
        {
            "person_name",
            "first_name",
            "last_name",
            "birth_date",
            "email",
            "phone_number",
            "tax_id",
            "identity_card",
            "driver_license",
            "passport",
            "credit_card",
            "bank_account",
            "street_address",
            "postal_code",
            "ip_address",
            "mac_address",
            "username",
            "gender",
            "religious_belief",
            "political_opinion",
            "trade_union",
            "ethnicity_or_race",
            "health_or_medical",
            "sexual_orientation",
            "genetic_or_biometric",
        }
    )
    profiles: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "identifier": (
                "code sequence id pk primary key sku matricula record_id index seq",
                "column: id cod_item id_pedido codigo sku uuid guid",
                "1 2 3 4 10 101 a1b2c3d4 9999 550e8400-e29b-41d4-a716-446655440000",
            ),
            "person_name": (
                "nome de pessoa person full name nome completo nome cliente usuario funcionario colaborador titular",
                "column: nome name full_name paciente solicitante passageiro autor cooperador",
                "joao silva maria oliveira carlos santos ana souza roberto almeida juliana lima",
            ),
            "first_name": (
                "primeiro nome first name given name prenome apelido",
                "column: primeiro_nome first_name fname prenome apelido",
                "joao maria carlos ana roberto lucas juliana gabriel beatriz pedro",
            ),
            "last_name": (
                "sobrenome last name surname family name",
                "column: sobrenome last_name surname familia",
                "silva santos oliveira souza lima pereira almeida ferreira alves",
            ),
            "birth_date": (
                "data de nascimento birth date birthday date of birth aniversario dt_nasc",
                "column: data_nascimento birth_date dt_nascimento data_nasc",
                "1990-05-20 15/03/1985 01/12/2000 1998/10/11 25-11-1976",
            ),
            "email": (
                "email address electronic mail email_address correio eletronico mailbox",
                "column: email email_address contato e_mail email_secundario",
                "usuario@email.com contato@empresa.com.br dev.user@gmail.com financeiro@dominio.org",
            ),
            "phone_number": (
                "telefone celular phone telephone mobile cell phone ddd ddi whatsapp fone",
                "column: telefone celular phone mobile tel cel whatsapp contato_fone fone_comercial",
                "+55 11 98765-4321 (21) 9988-7766 +1 415 555 2671 11988887777 (31) 3222-1100",
            ),
            "tax_id": (
                "cpf cnpj cadastro pessoa fisica juridica fiscal vat tin tax code ssn",
                "column: cpf cnpj doc_fiscal nr_cpf nr_cnpj tax_id ssn vat_number",
                "123.456.789-00 12.345.678/0001-90 98765432100 00.000.000/0001-91 123-45-6789",
            ),
            "identity_card": (
                "rg documento identidade general registry national identity card cedula",
                "column: rg documento_identidade ident rne cnh_rg cedula_identidade",
                "12.345.678-9 9876543-x 45.678.901-2 mg-12.345.678 sp-44.111.222-3",
            ),
            "driver_license": (
                "cnh habilitacao carteira nacional de motorista drivers license dlv",
                "column: cnh nr_cnh carteira_motorista habilitacao driver_license",
                "12345678901 00987654321 44556677889 99887766554",
            ),
            "passport": (
                "passaporte passport number documento internacional",
                "column: passaporte passport passport_number num_passaporte",
                "cs123456 ab987654 f7654321 br123987",
            ),
            "credit_card": (
                "cartao de credito credit card pan card number cvv expiration data bandeira",
                "column: num_cartao card_number credit_card cartao numero_cartao pan",
                "4532 1234 5678 9010 5412-7512-3412-3456 4000123456789010 378282246310005",
            ),
            "bank_account": (
                "conta bancaria agencia digito banco routing number iban swift bank account",
                "column: agencia conta num_conta bank_account conta_corrente iban swift chave_pix",
                "ag: 1234 cc: 12345-6 0001/01020304-5 br12345678901234567890123 033-1122-3344",
            ),
            "street_address": (
                "endereco logradouro rua avenida bairro numero complemento address street avenue road",
                "column: endereco logradouro rua avenue street address bairro complemento moradia",
                "av paulista 1000 apto 42 rua das flores 123 5th avenue suite 400 alameda santos 850",
            ),
            "postal_code": (
                "cep codigo postal postal code zip code zipcode",
                "column: cep zip_code postal_code cod_postal zip",
                "01310-100 04571-010 90210 10001-0001 30140-071",
            ),
            "location_city": (
                "cidade municipio localidade municipality city town",
                "column: cidade city municipio localidade",
                "sao paulo rio de janeiro curitiba belo horizonte salvador new york chicago tokyo",
            ),
            "location_state": (
                "estado provincia uf federacao unit state province",
                "column: estado state uf provincia unidade_federativa",
                "sp rj mg pr rs sc ba california texas florida",
            ),
            "location_country": (
                "pais nacao territorio country nation",
                "column: pais country nation patria",
                "brasil brasilian united states eua portugal argentina germany canada",
            ),
            "ip_address": (
                "endereco ip internet protocol ipv4 ipv6 host address",
                "column: ip ip_address client_ip host_ip remote_addr",
                "192.168.1.1 10.0.0.1 172.16.254.1 200.189.12.44 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
            ),
            "mac_address": (
                "mac address hardware address physical address endereco fisico",
                "column: mac mac_address ethernet bssid",
                "00:1a:2b:3c:4d:5e 00-14-22-01-23-45 01:23:45:67:89:ab",
            ),
            "username": (
                "nome de usuario username login user handle nickname arroba",
                "column: username login user_name usuario alias apelido",
                "admin user123 j_silva root dev_master maria_tech ana_99",
            ),
            "gender": (
                "genero sexo biologico gender sex identidade de genero",
                "column: genero gender sexo sex id_genero",
                "masculino feminino homem mulher male female m f non-binary travesti trans",
            ),
            "ethnicity_or_race": (
                "raca etnia cor autodeclaracao ethnicity race ancestry cor_raca",
                "column: raca etnia race cor autodeclaracao_raca cor_pele",
                "branca preta parda amarela indigena caucasian black asian afrodescendente",
            ),
            "religious_belief": (
                "religiao crenca religiosa confissao religiosa fe culto faith religion creed church",
                "column: religiao creed faith crenca culto confissao_religiosa denominacao",
                "catolica evangelica espirita candomble umbanda judaica islamica budista ateu agnostico protestante",
            ),
            "political_opinion": (
                "opiniao politica partido politico ideologia politica posicionamento politico filiacao partidaria political view party",
                "column: partido partido_politico ideologia orientacao_politica filiacao_partidaria",
                "pt pl psdb psol uniao brasil republicanos esquerda direita centro liberal conservador",
            ),
            "trade_union": (
                "filiacao sindical sindicato entidade sindical associacao profissional trade union labor union",
                "column: sindicato entidade_sindical filiacao_sindical orgao_classe associacao",
                "sindipetro cut forca sindical ugt oab crm cda filiado sindicalizado",
            ),
            "health_or_medical": (
                "saude historico medico doenca diagnostico cid prescricao exame health medical condition prescription prontuario",
                "column: doenca diagnostico cid prescricao laudo resultado_exame patologia alergia prontuario tratamento",
                "hipertensao diabetes tipo 2 cid-10 i10 covid-19 fratura asma positivo negativo cancer cardiomiopatia",
            ),
            "sexual_orientation": (
                "orientacao sexual vida sexual sexual orientation sex life",
                "column: orientacao_sexual vida_sexual sexual_orientation",
                "heterossexual homossexual bissexual assexual pansexual lgbtqia gay lesbica",
            ),
            "genetic_or_biometric": (
                "dado genetico dado biometrico biometria dna sequenciamento impressao digital reconhecimento facial iris",
                "column: biometria dna dado_genetico digital_hash facial_vector fingerprint iris",
                "hash_biometrico fingerprint_data base64_template locus_d8s1179 alelo sequence_vcf",
            ),
            "monetary_value": (
                "valor preco custo total salario pagamento dinheiro preco unitario price amount expense",
                "column: valor valor_compra preco preco_unitario total_pago amount price faturamento",
                "150.00 29.90 1250.50 10.00 99.99 4500.00 0.99 12.50 340.00 45.90",
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class FlowchartNode:
    id: str
    label: str
    shape: str
    box: tuple[int, int, int, int]
    center: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FlowchartEdge:
    source: str
    target: str
    direction: str = "down"


@dataclass(frozen=True, slots=True)
class FlowchartGraph:
    nodes: Sequence[FlowchartNode]
    edges: Sequence[FlowchartEdge]
    mermaid: str


@dataclass(frozen=True, slots=True)
class FlowchartInput:
    image: np.ndarray
    lang: str = "eng"


@dataclass(frozen=True, slots=True)
class FlowchartConfig:
    min_area: int = 1500
    threshold_value: int = 210
    tesseract_config: str = "--psm 6"
    padding: int = 4
    polygon_epsilon_ratio: float = 0.03


@dataclass(frozen=True, slots=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass(frozen=True, slots=True)
class DetectedTable:
    label: str
    score: float
    box: BoundingBox


@dataclass(frozen=True, slots=True)
class TableDetectionInput:
    image: Image.Image
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class TableDetectionConfig:
    model_name: str = "microsoft/table-transformer-detection"
    device: str | None = None
    default_threshold: float = 0.7
