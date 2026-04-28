#!/usr/bin/env python3
"""
Create IEEE conference paper Word document from template.
"""

import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree

TEMPLATE = '/home/runner/work/ECGAnomaliDetection/ECGAnomaliDetection/conference-template-a4 (1).docx'
OUTPUT   = '/home/runner/work/ECGAnomaliDetection/ECGAnomaliDetection/ECG_CNN_VAE_IEEE.docx'
IMG_DIR  = '/tmp/pdf_images'

# ── Image paths ───────────────────────────────────────────────────────────────
FIGS = {
    1:  (os.path.join(IMG_DIR, 'page2_img2_857x318.png'),    3.3),
    2:  (os.path.join(IMG_DIR, 'page2_img3_1245x395.png'),   3.3),
    3:  (os.path.join(IMG_DIR, 'page4_img10_989x790.png'),   3.3),
    4:  (os.path.join(IMG_DIR, 'page4_img11_815x624.png'),   3.3),
    5:  (os.path.join(IMG_DIR, 'page5_img12_658x547.png'),   3.3),
    6:  (os.path.join(IMG_DIR, 'page5_img13_592x547.png'),   3.3),
    7:  (os.path.join(IMG_DIR, 'page5_img14_522x470.png'),   3.3),
    8:  (os.path.join(IMG_DIR, 'page6_img15_634x547.png'),   3.3),
    9:  (os.path.join(IMG_DIR, 'page6_img16_985x547.png'),   3.3),
    10: (os.path.join(IMG_DIR, 'page6_img17_1274x430.png'),  3.3),
}
EQS = {
    1: (os.path.join(IMG_DIR, 'page3_img4_416x80.png'),   2.0),
    2: (os.path.join(IMG_DIR, 'page3_img5_454x110.png'),  2.0),
    3: (os.path.join(IMG_DIR, 'page3_img6_620x108.png'),  2.5),
    4: (os.path.join(IMG_DIR, 'page3_img7_316x116.png'),  1.5),
    5: (os.path.join(IMG_DIR, 'page3_img8_522x82.png'),   2.0),
    6: (os.path.join(IMG_DIR, 'page3_img9_166x54.png'),   1.0),
}

# ── XML helpers ───────────────────────────────────────────────────────────────
def _make_run(text, bold=False):
    r = OxmlElement('w:r')
    if bold:
        rPr = OxmlElement('w:rPr')
        b   = OxmlElement('w:b')
        rPr.append(b)
        r.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    r.append(t)
    return r


def insert_para_before(ref_elem, style_val, text='', centered=False, bold_prefix=''):
    """Insert a paragraph element immediately before ref_elem."""
    p = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    pStyle = OxmlElement('w:pStyle')
    pStyle.set(qn('w:val'), style_val)
    pPr.append(pStyle)
    if centered:
        jc = OxmlElement('w:jc')
        jc.set(qn('w:val'), 'center')
        pPr.append(jc)
    p.append(pPr)
    if bold_prefix:
        p.append(_make_run(bold_prefix, bold=True))
    if text:
        p.append(_make_run(text))
    ref_elem.addprevious(p)
    return p


def insert_image_before(doc, ref_elem, image_path, width_in, centered=True):
    """Insert an inline image paragraph immediately before ref_elem."""
    if not os.path.exists(image_path):
        print(f'  WARNING: image not found: {image_path}')
        insert_para_before(ref_elem, 'Body Text', f'[IMAGE MISSING: {os.path.basename(image_path)}]', centered=centered)
        return
    p = doc.add_paragraph()
    if centered:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(image_path, width=Inches(width_in))
    # Detach from the document body and re-attach before ref_elem
    p._element.getparent().remove(p._element)
    ref_elem.addprevious(p._element)
    return p


def insert_fig(doc, ref_elem, fig_num, caption):
    path, w = FIGS[fig_num]
    insert_image_before(doc, ref_elem, path, w, centered=True)
    insert_para_before(ref_elem, 'figure caption', caption, centered=True)


def insert_eq(doc, ref_elem, eq_num):
    path, w = EQS[eq_num]
    insert_image_before(doc, ref_elem, path, w, centered=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def build():
    doc  = Document(TEMPLATE)
    body = doc.element.body
    children = list(body)  # direct children (paragraphs + final sectPr)

    print(f'Template direct children: {len(children)}')

    # ── 1. Update title (para[0]) ─────────────────────────────────────────────
    p0 = children[0]
    for r in p0.findall(qn('w:r')):
        p0.remove(r)
    for r in p0.findall('.//' + qn('w:r')):
        p0.remove(r)
    p0.append(_make_run('ECG Anomali Tespiti için CNN-VAE ve Ansamble Derin Öğrenme Yaklaşımı'))

    # ── 2. Author block: clear paras[1-9], set simple text on para[4] ─────────
    for idx in range(1, 10):
        p = children[idx]
        for r in list(p.findall('.//' + qn('w:r'))):
            r.getparent().remove(r)

    author_lines = [
        'Yazar Adı Soyadı',
        'Bilgisayar Mühendisliği Bölümü',
        'Üniversite Adı',
        'Şehir, Türkiye',
        'email@example.com',
    ]
    p4 = children[4]
    for line in author_lines:
        r = _make_run(line)
        p4.append(r)
        br = OxmlElement('w:r')
        brEl = OxmlElement('w:br')
        br.append(brEl)
        p4.append(br)

    # ── 3. Abstract (para[13]) ────────────────────────────────────────────────
    p13 = children[13]
    for r in list(p13.findall('.//' + qn('w:r'))):
        r.getparent().remove(r)
    abstract_text = (
        'Abstract—Bu çalışmada, MIT-BIH Long-Term Arrhythmia veri tabanından elde edilen tek kanallı (MLII) '
        'elektrokardiyogram (ECG) sinyalleri üzerinde otomatik anomali tespiti gerçekleştirilmiştir. [2] '
        'Ham sinyal öncelikle 5–50 Hz bant geçiren Butterworth filtresi ile gürültüden arındırılmış, ardından '
        'Pan–Tompkins yaklaşımına benzer bir türev–kare alma–kayan ortalama yapısı kullanılarak R–peak\'ler '
        'tespit edilmiştir. Her bir R–peak etrafında 0.2 s öncesi ve 0.4 s sonrası olacak şekilde sabit '
        'uzunlukta beat segmentleri çıkarılmış; fizyolojik RR aralıkları, genlik varyansı ve enerji tabanlı '
        'ek maskeleme adımları ile yalnızca istatistiksel olarak "normal" kabul edilen beat\'ler eğitim için '
        'seçilmiştir. Ana model olarak, 1-B konvolüsyon katmanları ile ECG beat\'lerinden latent temsil '
        'öğrenen bir CNN-VAE mimarisi tasarlanmıştır. Bu modelin rekonstrüksiyon hatası, anomali skoru olarak '
        'kullanılmış ve veri kümesi üzerinde AUC = 1.00 değeri elde edilmiştir. Karşılaştırma amacıyla LSTM '
        'autoencoder, RNN autoencoder, latent uzay üzerinde GMM, özellik vektörleri üzerinde Isolation Forest '
        've VAE + LSTM + GMM skorlarını birleştiren bir ensemble model uygulanmıştır. Ensemble model, '
        'AUC ≈ 0.99 ile ikinci en yüksek performansı göstermiş, özellikle yanlış pozitif sayısını azaltarak '
        'daha dengeli bir performans ortaya koymuştur. Sonuçlar, CNN-VAE tabanlı yaklaşımın ECG beat\'lerinin '
        'morfolojisini oldukça başarılı biçimde yeniden inşa ettiğini ve küçük sayıda anomali içeren '
        'dengesiz veri senaryolarında bile yüksek ayrıştırma kapasitesi sunduğunu göstermektedir. Buna '
        'karşın LSTM ve RNN tabanlı autoencoder modelleri, hem yüksek yanlış alarm oranı hem de kaçan '
        'anomali sayısı bakımından zayıf kalmıştır. Çalışma, derin konvolüsyonel varyasyonel otoenkoderlerin, '
        'geleneksel zaman serisi modellere göre ECG anomali tespiti için daha uygun bir temel mimari olduğunu '
        'ortaya koymaktadır.'
    )
    p13.append(_make_run(abstract_text))

    # ── 4. Keywords (para[14]) ────────────────────────────────────────────────
    p14 = children[14]
    for r in list(p14.findall('.//' + qn('w:r'))):
        r.getparent().remove(r)
    p14.append(_make_run('Keywords—ECG, CNN-VAE, varyasyonel otoenkoder, anomali tespiti, derin öğrenme, aritmi, MIT-BIH'))

    # ── 5. Delete body sample paras [15 .. 92] ────────────────────────────────
    # Refresh children list after edits
    children = list(body)
    print(f'Children before deletion: {len(children)}')

    # Identify para[93] (the one with sectPr, after the references)
    # It is at index 93 of the current children list
    para93 = children[93]
    assert para93.find('.//' + qn('w:sectPr')) is not None, 'para[93] should have sectPr!'

    # Remove children[15] through children[92]
    to_remove = children[15:93]
    for elem in to_remove:
        body.remove(elem)

    print(f'Children after deletion: {len(list(body))}')

    # ── 6. Clear text from para[93] (now at a lower index), keep sectPr ───────
    children = list(body)
    # Find para93 by identity
    para93_idx = list(body).index(para93)
    print(f'para93 now at index {para93_idx}')

    # Remove only w:r elements (not pPr which contains sectPr)
    for r in list(para93.findall(qn('w:r'))):
        para93.remove(r)
    # Also remove any inline runs that might be nested in bookmarks etc.
    for r in list(para93.findall('.//' + qn('w:r'))):
        r.getparent().remove(r)

    # ── 7. Insert body content before para93 ──────────────────────────────────
    ref = para93  # all insertions go before this element

    def bp(text='', style='Body Text', centered=False, bold_prefix=''):
        insert_para_before(ref, style, text=text, centered=centered, bold_prefix=bold_prefix)

    def heading(text, level=1):
        style = 'Konu Başlığı' if level == 1 else ('Konu Başlığı 2' if level == 2 else 'Konu Başlığı 2')
        insert_para_before(ref, style, text=text)

    def bullet(text):
        insert_para_before(ref, 'bullet list', text=text)

    def caption(text):
        insert_para_before(ref, 'figure caption', text=text, centered=True)

    # ── I. GİRİŞ ──────────────────────────────────────────────────────────────
    heading('I. GİRİŞ', 1)

    bp('Kardiyak ritim bozukluklarının erken ve güvenilir bir şekilde tespiti, hem yoğun bakım ünitelerinde '
       'hem de uzun süreli holter kayıtlarında kritik öneme sahiptir. Elektrokardiyogram (ECG), kalbin '
       'elektriksel aktivitesini doğrudan yansıtan temel biyomedikal sinyal olup; QRS kompleksinin yapısı, '
       'RR aralıkları ve dalga morfolojileri üzerinden birçok patolojik duruma ilişkin ipucu barındırır. '
       'Uzun süreli kayıtların artmasıyla birlikte, bu sinyallerin manuel olarak incelenmesi hem zaman '
       'alıcı hale gelmiş, hem de gözden kaçan anomalilerin klinik riski gündeme gelmiştir. Bu nedenle, '
       'ECG sinyallerinde otomatik ve güvenilir anomali tespiti sağlayan yöntemlere duyulan ihtiyaç giderek '
       'artmaktadır.')

    bp('Son yıllarda, derin öğrenme tabanlı otoenkoder mimarileri, etiketli veri gereksinimini azaltmaları '
       've "normal" veriden yola çıkarak anomaliyi dolaylı olarak modelleyebilmeleri sebebiyle yoğun ilgi '
       'görmüştür. Özellikle varyasyonel otoenkoder (VAE) yapıları, girdiyi hem yeniden inşa etmeyi hem de '
       'latent uzayda olasılıksal bir temsil öğrenmeyi hedefleyerek, rekonstrüksiyon hatası ve latent '
       'yoğunluk bilgilerini aynı anda kullanmaya imkân tanır. Buna rağmen, zaman serisi alanında yaygın '
       'olarak kullanılan LSTM ve RNN tabanlı otoenkoderler ile konvolüsyon tabanlı VAE yaklaşımının uzun '
       'süreli ECG sinyallerindeki performanslarının sistematik biçimde karşılaştırıldığı çalışma sayısı '
       'sınırlıdır.')

    bp('Bu çalışmada amaç, MIT-BIH Long-Term Arrhythmia veri tabanından seçilen tek kanallı ECG kayıtları '
       'üzerinde, CNN-VAE tabanlı bir anomali tespit sistemi tasarlamak ve bu sistemi LSTM autoencoder, '
       'RNN autoencoder, GMM, Isolation Forest ve hibrit bir ensemble model ile karşılaştırmaktır. '
       'Çalışmanın katkıları şu şekilde özetlenebilir:')

    bullet('Uzun süreli ECG sinyalinden beat seviyesinde segment çıkaran, R–peak tabanlı ve çok aşamalı '
           'bir ön işleme ve normal beat seçimi pipeline\'ı oluşturulmuştur (band-pass filtreleme, R-peak '
           'tespiti, RR aralığı filtresi, genlik varyansı ve enerji tabanlı maskeleme).')
    bullet('Beat morfolojisini öğrenen 1-B CNN-VAE mimarisi tasarlanmış ve rekonstrüksiyon hatası ile '
           'latent uzay özellikleri anomali skoru olarak kullanılmıştır.')
    bullet('CNN-VAE\'nin performansı; LSTM AE, RNN AE, latent uzayda GMM, özellik uzayında Isolation '
           'Forest ve bu üç skoru birleştiren ensemble yaklaşım ile detaylı olarak karşılaştırılmıştır.')
    bullet('Tüm modeller için ROC eğrileri, AUC değerleri, confusion matrix\'ler, model karşılaştırma '
           'tablosu ve normalleştirilmiş performans heatmap\'i raporlanarak, pratikte hangi mimarinin '
           'hangi senaryoda daha avantajlı olduğu tartışılmıştır.')

    bp('Elde edilen sonuçlar, beat morfolojisini konvolüsyonel katmanlar ile öğrenen CNN-VAE mimarisinin, '
       'hem saf rekonstrüksiyon hatasıyla hem de hibrit ensemble içinde kullanıldığında, zaman serisi '
       'tabanlı modellere göre belirgin üstünlük sağladığını göstermektedir.')

    # ── II. YÖNTEM ─────────────────────────────────────────────────────────────
    heading('II. YÖNTEM', 1)

    heading('A. Veri Kümesi', 2)
    bp('Bu çalışmada kullanılan veri, MIT-BIH Long-Term Arrhythmia (LTA) veri tabanından alınmıştır. '
       'Bu veri tabanı, uzun süreli (yaklaşık 24 saat) ECG kayıtları içermekte olup aritmi türlerinin '
       'doğal dağılımını yansıtan gerçek klinik ortam kayıtlarından oluşur. Çalışmada yalnızca MLII '
       'derivasyonu kullanılmıştır. Sinyaller 360 Hz örnekleme frekansına sahiptir.')
    bp('Analiz iki seviyede yapılmıştır:')
    bullet('Sinyal seviyesi: Ham ECG\'nin filtrelenmesi ve R-peak tespiti')
    bullet('Beat seviyesi: Her R-peak etrafından sabit uzunlukta beat segmenti çıkarılması')
    bp('Model eğitimleri yalnızca normal beat\'ler üzerinde gerçekleştirilmiş; anomaliler, öğrenilen '
       'normal morfolojiden sapmalar üzerinden tespit edilmiştir.')

    insert_image_before(doc, ref, FIGS[1][0], FIGS[1][1], centered=True)
    caption('Şekil 1. Ham ECG sinyalinden örnek bir kalp atışı')

    heading('B. Sinyal Ön İşleme', 2)
    bp('Uzun süreli ECG kayıtlarında baz çizgisi kayması, kas artefaktı, güç hattı gürültüsü (50/60 Hz), '
       'hareket kaynaklı bozulmalar ve yüksek frekanslı istenmeyen bileşenler yaygındır. Bu nedenle ham '
       'sinyale aşağıdaki ön işleme adımları uygulanmıştır:')

    insert_para_before(ref, 'Body Text', text=' Ham sinyale 5–50 Hz aralığında, 4. dereceden Butterworth '
                       'bant geçiren filtre uygulanmıştır. Bu aralık, QRS kompleksinin temel enerji '
                       'dağılımının bulunduğu frekans bandına karşılık gelir ve düşük/hızlı bileşenler '
                       'bastırılmış olur.', bold_prefix='(a) Bant Geçiren Filtreleme: ')

    insert_para_before(ref, 'Body Text', text=' Filtrelenmiş sinyalde R–peak tespiti için Pan–Tompkins '
                       'yaklaşımına benzer bir yapı kullanılmıştır:', bold_prefix='(b) R-Peak Tespiti [1]: ')

    bullet('Türev alma')
    bullet('Kare alma')
    bullet('Kayan pencere entegrasyonu')
    bullet('Yerel maksimum arama')
    bp('Doğrulama amacıyla algılanan R–peak\'ler sinyal üzerinde işaretlenmiş ve tüm kayıt için yüksek '
       'doğruluk elde edilmiştir.')

    heading('C. Beat Segmentasyonu ve Normal Beat Seçimi', 2)
    bp('Her R–peak etrafında sabit uzunlukta beat çıkarmak için aşağıdaki pencere yapısı uygulanmıştır:')
    bullet('0.2 s önce')
    bullet('0.4 s sonra')
    bp('Bu, toplam 216 örneklik (≈ 600 ms) bir segment oluşturur. Tüm beat\'ler aynı uzunlukta olacak '
       'şekilde normalize edilmiştir.')

    insert_para_before(ref, 'Body Text', bold_prefix='Normal Beat Seçimi')
    bp('Autoencoder modelleri yalnızca normal beat\'lerle eğitildiği için, anormal beat\'lerin eğitim '
       'sürecine karışmaması kritik önem taşır. Bu amaçla aşağıdaki filtreleme kriterleri uygulanmıştır:')
    bullet('RR interval filtrelemesi: 0.4s < RR < 1.5')
    bullet('Genlik varyansı filtresi: Fiziksel olarak aşırı düşük veya yüksek varyanslı beat\'ler elenmiştir.')
    bullet('Enerji tabanlı maskeleme: Çok düşük enerjiye sahip "flattened" segmentler çıkarılmıştır.')
    bp('Bu işlemler sonucunda istatistiksel olarak tutarlı ve temiz normal beat\'lerden oluşan büyük bir '
       'eğitim kümesi elde edilmiştir.')

    insert_image_before(doc, ref, FIGS[2][0], FIGS[2][1], centered=True)
    caption('Şekil 2. Segmentasyon sonrası örnek bir beat')

    heading('D. Ölçekleme', 2)
    bp('Her beat, 216×1 boyutunda olup Min-Max normalizasyonu ile [0,1] aralığına ölçeklenmiştir. '
       'Bu hem model eğitim stabilitesini artırmakta hem de farklı beat\'lerin genlik farklarını '
       'normalize ederek öğrenmeyi kolaylaştırmaktadır.')

    heading('E. Kullanılan Modeller', 2)
    bp('Çalışmada beş temel model ve bir hibrit ansamble yaklaşımı uygulanmıştır.')

    # 1) CNN-VAE
    insert_para_before(ref, 'Konu Başlığı 2', text='1) CNN-VAE')
    bp('Ana model olarak tasarlanan CNN-VAE, beat morfolojisini hem konvolüsyonel uzamsal özelliklerle '
       'hem de varyasyonel latent temsille öğrenmektedir.')
    insert_para_before(ref, 'Body Text', bold_prefix='Encoder:')
    bullet('Conv1D (32 filtre, kernel=7)')
    bullet('MaxPooling')
    bullet('Conv1D (64 filtre)')
    bullet('MaxPooling')
    bullet('Global Average Pooling')
    bullet('Conv1D')
    insert_eq(doc, ref, 1)

    insert_para_before(ref, 'Body Text', bold_prefix='Reparametrization Trick:')
    insert_eq(doc, ref, 2)

    insert_para_before(ref, 'Body Text', bold_prefix='Decoder:')
    bullet('Dense → yeniden şekillendirme')
    bullet('UpSampling1D')
    bullet('Conv1D katmanları ile beat rekonstrüksiyonu')

    insert_para_before(ref, 'Body Text', bold_prefix='Kayıp Fonksiyonu:')
    insert_eq(doc, ref, 3)
    bp('(beta=0.001 olarak seçildi.) Bu model çalışmanın en yüksek performansını göstermiştir.')

    # 2) LSTM AE
    insert_para_before(ref, 'Konu Başlığı 2', text='2) LSTM Autoencoder')
    bp('Beat zaman serisi doğrudan ardışık örnekler olarak işlenmiştir.')
    bullet('Encoder: LSTM(64) → LSTM(32)')
    bullet('Decoder: LSTM(64) → Dense(1)')
    bp('LSTM AE, zaman bağımlılığını yakalayabilse de morfolojik uzaysal yapıları CNN kadar iyi '
       'öğrenemediği gözlenmiştir.')

    # 3) RNN AE
    insert_para_before(ref, 'Konu Başlığı 2', text='3) RNN Autoencoder')
    bp('LSTM yerine klasik RNN hücreleri (SimpleRNN) kullanılmıştır.')
    bullet('Encoder: RNN(64) → RNN(32)')
    bullet('Decoder: RNN(64) → Dense(1)')
    bp('Zaman serisi modellemesine yatkın olmasına rağmen rekonstrüksiyon kabiliyeti sınırlı kalmıştır.')

    # 4) GMM
    insert_para_before(ref, 'Konu Başlığı 2', text='4) GMM (Gaussian Mixture Model)')
    bp('CNN-VAE encoder\'ından elde edilen 16 boyutlu latent vektörler üzerinde GMM uygulanmıştır.')
    insert_para_before(ref, 'Body Text', bold_prefix='Anomali skoru:')
    insert_eq(doc, ref, 4)
    bp('Yani latent alandaki düşük olasılıklı örnekler anomalidir.')

    # 5) Isolation Forest
    insert_para_before(ref, 'Konu Başlığı 2', text='5) Isolation Forest')
    bp('Beat\'ler önce 216 boyutlu vektörlere flatten edilmiştir. Daha sonra:')
    bullet('200 ağaçlı Isolation Forest eğitilmiştir.')
    bullet('Skor olarak izolasyon derinliği kullanılmıştır.')
    bp('Bu model, morfolojik benzerlikleri güçlü şekilde yakalayamadığı için düşük performans '
       'göstermiştir. [6]')

    # 6) Ensemble
    insert_para_before(ref, 'Konu Başlığı 2', text='6) Ensemble Model (VAE + LSTM + GMM)')
    bp('Daha dengeli bir model oluşturmak için üç farklı skor birleştirilmiştir:')
    insert_eq(doc, ref, 5)
    bp('Sonrasında:')
    bullet('Normal dağılım varsayımıyla dinamik eşik:')
    insert_eq(doc, ref, 6)
    bp('Bu eşiğin üzerindeki beat\'ler anomali olarak etiketlenmiştir. Ensemble model, yanlış '
       'pozitifleri azaltmış ve CNN-VAE\'den sonra ikinci en iyi genel performansı göstermiştir.')

    # ── III. DENEYSEL SONUÇLAR ─────────────────────────────────────────────────
    heading('III. DENEYSEL SONUÇLAR', 1)
    bp('Bu bölümde, çalışmada kullanılan tüm modellerin performansı ayrıntılı şekilde değerlendirilmiş, '
       'ROC eğrileri, hata dağılımları, ansamble model analizi ve her model için konfüzyon matrisleri '
       'sunulmuştur. Ayrıca tüm modeller arasındaki karşılaştırma tablo ve ısı haritası ile özetlenmiştir.')

    heading('A. Rekonstrüksiyon Hatalarının Analizi', 2)
    bp('Autoencoder tabanlı modellerde anomali tespiti temel olarak rekonstrüksiyon hatasına dayanır. '
       'Normal beat\'lerde hata düşük, anomalilerde belirgin derecede yüksektir. Bu davranış özellikle '
       'CNN-VAE modelinde çok net gözlenmiştir. Rekonstrüksiyon hatasının dağılım grafikleri (Şekil 3) '
       'incelendiğinde:')
    bullet('CNN-VAE → Anomali ile normal beat arasındaki ayrım en keskin modeldir.')
    bullet('LSTM Autoencoder → Ayrım daha az belirgin olup normal/anomali dağılımları kısmen üst üste '
           'binmektedir.')
    bullet('RNN Autoencoder → Zaman bağımlılığını yakalamasına rağmen morfolojik varyasyonlara karşı '
           'zayıf kalmıştır.')
    bp('Bu sonuç, CNN tabanlı morfolojik öğrenme yapılarının ECG beat analizinde belirgin bir avantaj '
       'sağladığını göstermektedir.')
    insert_image_before(doc, ref, FIGS[3][0], FIGS[3][1], centered=True)
    caption('Şekil 3. Latent uzayında normal ve anomali beat dağılımı')

    heading('B. ROC Eğrisi Karşılaştırmaları', 2)
    bp('Çalışmada tüm modeller için tek grafik üzerinde ROC eğrisi çizilmiştir. Elde edilen AUC değerleri:')
    bullet('CNN-VAE: 1.000')
    bullet('Ensemble (VAE + LSTM + GMM): 0.997')
    bullet('Isolation Forest: 0.845')
    bullet('GMM (Latent): 0.832')
    bullet('LSTM Autoencoder: 0.833')
    bullet('RNN Autoencoder: 0.826')
    bp('ROC eğrisi grafiği (Şekil 4), CNN-VAE modelinin mükemmele yakın ayrım gücüne sahip olduğunu '
       'açıkça göstermektedir. Ensemble model ROC eğrisi de model kararlılığını artırarak yüksek AUC '
       'değerine ulaşmıştır. Diğer modeller ise orta düzey ayırt edicilik göstermiştir.')
    insert_image_before(doc, ref, FIGS[4][0], FIGS[4][1], centered=True)
    caption('Şekil 4. Modellerin ROC eğrileri')

    heading('C. Modellere Ait Konfüzyon Matrisleri', 2)
    bp('Bu çalışmada konfüzyon matrisleri yalnızca test kümesindeki normal ve anomali sınıflarının '
       'ayrımı üzerinde değerlendirilmiştir.')
    bp('(a) CNN-VAE')
    insert_image_before(doc, ref, FIGS[5][0], FIGS[5][1], centered=True)
    caption('Şekil 5. CNN konfüzyon matrisi')

    bp('(b) LSTM Autoencoder')
    insert_image_before(doc, ref, FIGS[6][0], FIGS[6][1], centered=True)
    caption('Şekil 6. LSTM Autoencoder modeline ait konfüzyon matrisi')

    bp('(c) Ensemble Model (VAE + LSTM + GMM)')
    bp('CNN-VAE modeli hiçbir yanlış sınıflama yapmamıştır. Hem yanlış pozitif hem yanlış negatif '
       'oranı sıfırdır. Bu, modelin hem rekonstrüksiyon kapasitesinin hem de latent uzay temsilinin '
       'oldukça başarılı olduğunu göstermektedir.')
    insert_image_before(doc, ref, FIGS[7][0], FIGS[7][1], centered=True)
    caption('Şekil 7. Ensemble konfüzyon matrisi')

    bp('LSTM Autoencoder, zaman bağımlılığını modellemek konusunda iyi olsa da morfolojik anomalileri '
       'CNN yapısı kadar iyi temsil edemediğinden, hem yanlış pozitif hem yanlış negatif oranları daha '
       'yüksektir.')
    bp('Ensemble model, CNN-VAE kadar kusursuz olmasa da:')
    bullet('Yanlış pozitifleri azaltmış')
    bullet('Yanlış negatifleri CNN-VAE dışındaki tüm modellerden daha iyi kontrol etmiş')
    bp('Bu nedenle ansamble, stabilite açısından en iyi ikinci model olarak değerlendirilmiştir.')

    heading('D. Isı Haritası ile Model Karşılaştırma', 2)
    bp('Modellerin AUC, ortalama rekonstrüksiyon hatası ve tespit edilen anomali sayısı metrikleri '
       'normalize edilerek tek bir ısı haritasında gösterilmiştir (Şekil 8). '
       'Isı haritası analizi şu sonuçları ortaya koymaktadır:')
    bullet('AUC açısından: CNN-VAE ≫ Ensemble > diğer modeller')
    bullet('Rekonstrüksiyon hatası açısından: Ensemble model (0.04) en düşük hata değerine sahiptir.')
    bullet('Anomali tespit gücü açısından: GMM latent modeli en fazla anomaliyi işaretlemesine rağmen '
           'yanlış pozitif oranı çok yüksektir.')
    bullet('Genel performansta: CNN-VAE = en yüksek doğruluk, Ensemble = en dengeli ve en kararlı model')
    insert_image_before(doc, ref, FIGS[8][0], FIGS[8][1], centered=True)
    caption('Şekil 8. Modellerin performans karşılaştırmasına ait ısı haritası')

    heading('E. Modellerin Genel Karşılaştırma Tablosu', 2)
    bp('Önemli gözlemler (Şekil 10):')
    bullet('CNN-VAE: En yüksek ayırt edicilik gücü')
    bullet('Ensemble: En düşük hata + yüksek doğruluk kombinasyonu')
    bullet('IF / GMM: Anomali sayısı yüksek fakat yanlış pozitif çok')
    bullet('RNN / LSTM: Orta düzey performans')
    insert_image_before(doc, ref, FIGS[9][0], FIGS[9][1], centered=True)
    caption('Şekil 9. LSTM ve İsolation Forest modellerinin kıyaslanması')
    insert_image_before(doc, ref, FIGS[10][0], FIGS[10][1], centered=True)
    caption('Şekil 10. Modellerin genel kıyaslanması')

    heading('F. Genel Değerlendirme', 2)
    bp('Bu çalışmada hem derin öğrenme tabanlı autoencoder yapıları hem de istatistiksel/anomali tespit '
       'algoritmaları bir arada değerlendirilmiştir. Sonuçlar, ECG beat morfolojisini tanımada '
       'konvolüsyonel mimarilerin (CNN-VAE) açık ara üstün olduğunu göstermektedir.')
    bullet('CNN-VAE modeli test kümesinde %100 doğruluk ile anomalileri tespit etmiştir.')
    bullet('Ensemble model, farklı yapıların güçlü yönlerini birleştirerek daha stabil ve düşük hatalı '
           'bir sonuç üretmiştir.')
    bullet('LSTM ve RNN tabanlı modeller zaman bilgisini iyi işlese de morfolojik detayları CNN kadar '
           'kuvvetli yakalayamamıştır.')
    bullet('GMM ve Isolation Forest gibi "unsupervised classical" yaklaşımlar temel seviyede performans '
           'göstermiştir.')
    bp('Genel olarak, CNN tabanlı varyasyonel autoencoder yaklaşımı bu tür uzun süreli ECG anomali '
       'tespiti görevlerinde en etkili yöntem olarak öne çıkmıştır.')

    # ── IV. TARTIŞMA ───────────────────────────────────────────────────────────
    heading('IV. TARTIŞMA', 1)
    bp('Bu çalışmada, uzun süreli MIT-BIH aritmi kayıtlarından elde edilen beat seviyesinde ECG '
       'sinyallerini kullanarak beş farklı model ve bir hibrit ansamble yapı karşılaştırılmıştır. '
       'Elde edilen sonuçlar, beat morfolojisinin yapısal özelliklerini öğrenmede konvolüsyon tabanlı '
       'autoencoder modellerinin belirgin bir avantaj sunduğunu göstermektedir. CNN-VAE modelinin '
       'rekonstrüksiyon hatası dağılımları son derece keskin olup, normal ve anormal beat\'ler arasında '
       'yüksek seviyede ayrım sağlamıştır.')

    bp('CNN-VAE\'nin başarısının temel nedenleri aşağıdaki şekilde özetlenebilir:')
    bp('1. Uzaysal morfoloji öğrenimi: QRS kompleksi ve çevresindeki dalga formlarının yapısı, zaman '
       'bağımlılığı kadar uzamsal filtreleme gerektirir. Conv1D katmanları bu yapıyı LSTM ve RNN\'e göre '
       'daha etkili yakalamıştır.')
    bp('2. Latent uzayda olasılıksal temsil: VAE\'nin sunduğu z_μ ve z_{log σ²} yapısı, modelin '
       'belirsizliği öğrenmesini ve düşük yoğunluklu örnekleri daha iyi ayırt etmesini sağlamıştır.')
    bp('3. Düşük overfitting eğilimi: CNN-VAE modeli nispeten küçük bir latent boyut (16) ve güçlü '
       'düzenleme etkisi olan KL-divergence ile eğitildiğinden, veriye aşırı uyum sağlamamıştır.')

    bp('Buna karşılık, LSTM ve RNN tabanlı autoencoder modelleri zaman serisi özelliklerini başarılı '
       'bir şekilde yakalasa da, beat morfolojisinin ayrım gerektiren ince detaylarını CNN tabanlı '
       'modeller kadar etkin şekilde modelleyememiştir. Bu durum özellikle yüksek yanlış pozitif '
       'oranları (FP) ve yüksek yanlış negatif oranları (FN) ile kendini göstermiştir. Klinik '
       'uygulamalarda yanlış negatiflerin kabul edilemez olması göz önüne alındığında, RNN tabanlı '
       'modellerin performansı pratik kullanım açısından yetersiz kalmaktadır.')

    bp('Isolation Forest ve GMM gibi klasik anomalilik yöntemleri, özellikle latent özellikler üzerinde '
       'çalıştıklarında belirli düzeyde performans göstermelerine rağmen, ECG gibi ince yapılı '
       'biyomedikal sinyallerde derin öğrenme tabanlı modellerin ayrıştırma gücüne ulaşamamıştır. '
       'Bunun temel nedenleri:')
    bullet('GMM\'in karmaşık morfolojik varyasyonları modellemek için yeterli ifade kapasitesine sahip '
           'olmaması,')
    bullet('Isolation Forest\'ın ise yüksek boyutlu zaman serisi uzayında ayrım yapmakta zorlanmasıdır.')

    bp('Ensemble model, farklı modellerin güçlü yönlerini birleştirerek daha dengeli bir performans '
       'sergilemiştir. CNN-VAE\'nin rekonstrüksiyon gücü, GMM\'in latent uzay yoğunluk tahmini ve '
       'LSTM\'in zaman bağımlılığını modelleme kapasitesi birleştiğinde, hem düşük hata hem de yüksek '
       'AUC değeri elde edilmiştir. Ancak CNN-VAE\'nin tek başına gösterdiği mükemmel performans, '
       'ansamble yapının dahi üzerine çıkmaktadır. Bu sonuçlar, derin konvolüsyonel VAE mimarisinin '
       'ECG beat anomali tespiti için özellikle uygun olduğunu ve klinik kullanım potansiyelinin '
       'yüksek olduğunu göstermektedir.')

    heading('A. Kısıtlar', 2)
    bp('Bu çalışmada önerilen CNN-VAE ve hibrit ansamble yaklaşımı umut verici sonuçlar ortaya koysa '
       'da, çalışmanın değerlendirilmesinde göz önünde bulundurulması gereken bazı kısıtlar '
       'mevcuttur:')

    insert_para_before(ref, 'Body Text',
                       text=' Çalışmada kullanılan test kümesinde, gerçek klinik senaryoları simüle '
                       'etmek amacıyla doğal bir sınıf dengesizliği korunmuştur. Ancak test edilen '
                       'anomali sayısının (N=20) normal beat sayısına (N=1796) oranla oldukça düşük '
                       'olması, elde edilen %100 doğruluk (AUC=1.00) değerinin istatistiksel '
                       'genelleme kapasitesi üzerinde bir belirsizlik yaratmaktadır. Daha geniş ve '
                       'çeşitli anomali türlerini içeren test kümeleriyle yapılacak doğrulamalar, '
                       'modelin güvenilirliğini artıracaktır.',
                       bold_prefix='Test Kümesindeki Dengesizlik ve Örneklem Boyutu: ')

    insert_para_before(ref, 'Body Text',
                       text=' Çalışma, MIT-BIH veri tabanından alınan tek kanallı (MLII derivasyonu) '
                       'kayıtlar üzerine kurgulanmıştır. Bazı kardiyak aritmilerin tespiti, çoklu '
                       'derivasyonların (12-lead ECG) eş zamanlı analizini gerektirebilir. Dolayısıyla '
                       'önerilen yöntem, tek kanalda belirti vermeyen patolojileri yakalamakta sınırlı '
                       'kalabilir.',
                       bold_prefix='Tek Kanallı Analiz: ')

    insert_para_before(ref, 'Body Text',
                       text=' Modelin başarısı, büyük ölçüde "normal" beat seçiminde kullanılan katı '
                       'ön işleme kurallarına (RR aralığı ve varyans filtreleri) dayanmaktadır. Farklı '
                       'gürültü seviyelerine sahip veya farklı cihazlardan alınan ham verilerde, bu ön '
                       'işleme adımlarının aynı hassasiyetle çalışmaması model performansını '
                       'etkileyebilir.',
                       bold_prefix='Ön İşleme Bağımlılığı: ')

    insert_para_before(ref, 'Body Text',
                       text=' Bu çalışma, temel olarak "Normal" ve "Anomali" ayrımı (binary '
                       'classification) üzerine odaklanmıştır. Anomalinin alt türlerinin (örneğin '
                       'Atriyal Fibrilasyon, PVC, LBBB vb.) sınıflandırılması bu çalışmanın kapsamı '
                       'dışında tutulmuştur.',
                       bold_prefix='Aritmi Türlerinin Sınıflandırılması: ')

    # ── V. SONUÇ ───────────────────────────────────────────────────────────────
    heading('V. SONUÇ', 1)
    bp('Bu çalışma, uzun süreli gerçek ECG kayıtlarında anomali tespiti için kapsamlı bir ön işleme '
       'pipeline\'ı ile birlikte birden fazla derin öğrenme ve geleneksel istatistiksel yöntemi '
       'değerlendirmiştir. Çalışmanın temel bulguları aşağıdaki şekilde özetlenebilir:')

    bp('1. CNN-VAE modeli, tüm modeller arasında en yüksek performansı sergilemiş olup, test kümesinde '
       'hem yanlış pozitif hem de yanlış negatif oranını sıfıra düşürerek %100 doğruluk elde etmiştir.')
    bp('2. Ensemble model, yüksek kararlılık sunmuş ve özellikle yanlış pozitif oranlarını düşürerek '
       'klinik açıdan daha dengeli bir sonuç sağlamıştır. AUC ≈ 0.99 ile ikinci en başarılı modeldir.')
    bp('3. LSTM ve RNN tabanlı autoencoder modelleri, ECG beat morfolojisinin inceliğini yakalamakta '
       'zorlanmış ve yüksek oranda yanlış alarm üretmiştir (FP ve FN değerlerinin yüksek olması).')
    bp('4. GMM latent modeli anomalileri agresif bir şekilde işaretleyerek en çok anomalinin tespit '
       'edilmesini sağlamış olsa da, yüksek yanlış pozitif oranı nedeniyle tek başına güvenilir '
       'değildir.')
    bp('5. Isolation Forest, klasik ML yöntemleri arasında en iyi performansı göstermiş olsa da, '
       'derin öğrenme modellerinin seviyesine ulaşamamıştır.')

    bp('Çalışma genel olarak göstermektedir ki: ECG beat anomali tespiti için konvolüsyon tabanlı '
       'varyasyonel autoencoder mimarisi (CNN-VAE), hem doğruluk hem de kararlılık açısından en uygun '
       'temel modeldir.')

    bp('Gelecek çalışmalar için aşağıdaki yönler değerlendirilebilir:')
    bullet('Çok kanallı veya 12 derivasyonlu ECG üzerinde modelin genişletilmesi,')
    bullet('VAE modeli için β-VAE, CVAE veya Diffusion VAE gibi alternatif varyasyonların test edilmesi,')
    bullet('Transfer learning ile farklı popülasyonlardaki ECG kayıtlarına adaptasyon,')
    bullet('Gerçek zamanlı inference yapacak hafifletilmiş modellerin tasarlanması.')

    bp('Bu çalışma, hem klinik araştırmacılar hem de makine öğrenimi topluluğu için ECG anomali '
       'tespitinde CNN-VAE tabanlı yaklaşımların potansiyelini göstermektedir. Özellikle CNN-VAE '
       'mimarisinin yüksek doğruluk, düşük hata oranı ve güçlü genelleme kabiliyeti, bu yöntemi hem '
       'araştırma hem de klinik uygulamalar için güvenilir bir temel model haline getirmektedir. '
       'Bulgular, doğru ön işleme pipeline\'ı ile desteklenen konvolüsyonel varyasyonel otoenkoderlerin, '
       'karmaşık biyomedikal sinyallerde morfolojik ayrımı yakalamada klasik ve tekrarlayan sinir ağı '
       'tabanlı modellere göre belirgin üstünlük sağladığını göstermektedir.')

    # ── KAYNAKÇA ───────────────────────────────────────────────────────────────
    insert_para_before(ref, 'Heading 5', text='References')

    refs = [
        '[1] J. Pan and W. J. Tompkins, "A real-time QRS detection algorithm," '
        'IEEE Transactions on Biomedical Engineering, vol. 32, no. 3, pp. 230–236, 1985.',
        '[2] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," '
        'IEEE Engineering in Medicine and Biology Magazine, vol. 20, no. 3, pp. 45–50, 2001.',
        '[3] D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," '
        'arXiv preprint arXiv:1312.6114, 2013.',
        '[4] Z. Zhao, H. Liu, J. Li, et al., "Anomaly detection using deep learning in ECG '
        'signals: A survey," Computers in Biology and Medicine, vol. 127, 2020.',
        '[5] F. Chollet, "Deep Learning with Python," Manning Publications, 2018.',
        '[6] L. Breiman, "Isolation Forest," Proceedings of the 8th IEEE ICDM, '
        'pp. 413–422, 2008.',
    ]
    for r_text in refs:
        insert_para_before(ref, 'references', text=r_text)

    # ── Clean up leftover template content (para[94]) ─────────────────────────
    children = list(body)
    # para[94] of original is now at some index after para93 – find by position
    # It's directly after para93; clear its w:r children
    para93_idx = list(body).index(para93)
    if para93_idx + 1 < len(children):
        p_after = children[para93_idx + 1]
        tag = p_after.tag.split('}')[-1]
        if tag == 'p':
            for r in list(p_after.findall('.//' + qn('w:r'))):
                r.getparent().remove(r)

    # ── Save ───────────────────────────────────────────────────────────────────
    doc.save(OUTPUT)
    print(f'\nSaved: {OUTPUT}')
    print(f'Size: {os.path.getsize(OUTPUT):,} bytes')


if __name__ == '__main__':
    build()
