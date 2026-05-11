import random

template_dict = {
    'Qwen2.5-7B': 'qwen',
    'Qwen2.5-14B': 'qwen',
    'Qwen2.5-32B': 'qwen',
    'Qwen2.5-72B': 'qwen',
    'Qwen2.5-7B-Instruct': 'qwen',
    'Qwen2.5-14B-Instruct': 'qwen',
    'Qwen2.5-32B-Instruct': 'qwen',
    'Qwen2.5-72B-Instruct': 'qwen',
    'Qwen3-14B': 'qwen3',
    'glm-4-9b': 'glm4',
    'glm-4-9b-chat': 'glm4',
    'GLM-4-9B-0414': 'glm4',
    'Llama3.1-8B-Chinese-Chat': 'llama3',
    'Meta-Llama-3.1-8B-Instruct': 'llama3',
    'Meta-Llama-3.1-8B': 'llama3',
}

llm_template = {
    "qwen": """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
""",
    "glm4": """[gMASK]<sop><|user|>
{}<|assistant|>
""",
    "llama3": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
}

stop_token_id_dict = {
    'qwen': 151645,
    'glm4': 151336,
    'llama3': 128009
}

stop_token_dict = {
    'qwen': '<|im_end|>',
    'glm4': '<|user|>'
}

pn_zh2en_source = """原文：萤灯姐姐	译文：Sister Yingdeng,
原文：姐姐升官大喜	译文：Congrats on your promotion,
原文：百事顺意	译文：may everything go your way.
原文：连姿容神气也远胜往日	译文：You even look more radiant than before.
原文：你我都出自帝君门下	译文：We both hail from Your Majesty's guidance,
原文：何须行礼	译文：no need for formalities between us.
原文：碧旖	译文：Biyi,
原文：姐姐自离宫后	译文：Since you left the palace,
原文：一早考取仙阶 升任仙倌	译文：you quickly climbed the ranks to become an immortal attendant,
原文：如今啊	译文：now,
原文：已是一阁之首	译文：leading a whole pavilion.
原文：可见这衍虚天宫众多仙侍	译文：It's clear among the many attendants in Yanxu Heavenly Palace,
原文：帝君对姐姐	译文：Your Majesty has a
原文：特别的不同啊	译文：special regard for you.
原文：帝君可在宫中	译文：Is Your Majesty around?
原文：此布面从装饰到镶嵌	译文：Every aspect of this fabric,
原文：都是萤灯掌事亲手所制	译文：is personally crafted by Yingdeng.
原文：混元玉带破损	译文：The Hunyuan Jade Belt was damaged,
原文：虽为无知仙侍颜淡所为	译文：though by the careless hands of Yandan,
原文：但妙法阁仍有失察之责	译文：Magical Pavilion still bears responsibility for the oversight.
原文：萤灯如今不能侍奉在侧	译文：Now that Yingdeng can't serve by your side,
原文：帝君常议事至天明	译文：You often work till dawn,
原文：唯愿这安神之物	译文：hopefully, this calming artifact
原文：能助帝君安眠	译文：can help you find peace in sleep.
原文：不必了	译文：No need for that.
原文：那仙侍虽无知鲁莽	译文：Though the immortal attendant is naive and reckless,
原文：但已经将玉带补好	译文：she's already repaired the jade belt.
原文：这布面美意过重	译文：This fabric's sentiment is too heavy,
原文：甚是铺张	译文：It's quite lavish,
原文：本君消受不起	译文：I can't accept such extravagance."""

pn_zh2en_identify = """萤灯（人名） - Yingdeng
帝君（称谓） - Your Majesty
碧旖（人名） - Biyi
衍虚天宫（机构/地名） - Yanxu Heavenly Palace
仙侍（职位） - immortal attendant
混元玉带（物品） - Hunyuan Jade Belt
颜淡（人名） - Yandan
妙法阁（机构） - Magical Pavilion"""

pn_zh2th_source = """原文：这个	译文：ยานี้
原文：换成款冬花	译文：ต้องเปลี่ยนเป็นควงตงฮวย
原文：陪你父亲好好说说话	译文：กลับไปคุยกับพ่อของเจ้าดี ๆ
原文：伯父都能跟你服软	译文：ขนาดลุงยังสามารถอ่อนให้เจ้าได้
原文：你就不能跟你父亲	译文：แล้วเจ้าไม่สามารถอ่อนให้กับ
原文：服个软吗	译文：พ่อของเจ้าเลยหรือ
原文：皇伯父	译文：เสด็จลุง...
原文：你要听话	译文：เจ้าต้องเชื่อฟังนะ
原文：是	译文：พ่ะย่ะค่ะ
原文：长青	译文：ฉางชิง
原文：奴婢在	译文：พ่ะย่ะค่ะ
原文：你觉得晏惜委屈吗	译文：เจ้าคิดว่าเยี่ยนซีอัดอั้นตันใจหรือไม่
原文：世子对凌王殿下	译文：รัฐทายาทคงยากที่จะให้อภัย
原文：恐怕难以释怀	译文：ท่านอ๋องหลิง
原文：当年凌王做下了这件事	译文：ตอนนั้นอ๋องหลิงทำเรื่องนี้ลงไป
原文：朕不戳破	译文：เราไม่เปิดโปง
原文：他便以为神不知鬼不觉	译文：เขาก็คิดว่าไม่มีผู้ใดรู้เรื่อง
原文：欠下的账也终归要还的	译文：หนี้ที่ติดค้างไว้ อย่างไรก็ต้องชดใช้
原文：何况	译文：อีกอย่าง
原文：晏惜是朕养大的	译文：เราเป็นคนเลี้ยงเยี่ยนซีจนโต
原文：如今晏惜受了委屈	译文：บัดนี้เยี่ยนซีต้องอัดอั้นตันใจ
原文：那朕就替他讨个公道	译文：เช่นนั้นเราก็จะทวงความยุติธรรมให้เขา
原文：这次	译文：ครั้งนี้
原文：父亲没回玉清山躲着	译文：ท่านพ่อไม่ได้ไปหลบอยู่ที่เขาอวี้ชิง
原文：司使忠于官家	译文：เสนาบดีจงรักภักดีต่อฝ่าบาท
原文：七宿卫便该忠于司使	译文：องครักษ์สัตตะดาราก็ควรจงรักภักดีต่อเสนาบดี
原文：京中之事	译文：เรื่องในเมืองหลวง
原文：七宿司无所不知	译文：ไม่มีเรื่องใดที่สำนักสัตตะดาราไม่รู้"""

pn_zh2th_identify = """款冬花（物品） - ควงตงฮวย
皇伯父（称谓） - เสด็จลุง
长青（人名） - ฉางชิง
晏惜（人名） - เยี่ยนซี
凌王（称谓） - อ๋องหลิง
朕（称谓） - เรา
玉清山（地名） - เขาอวี้ชิง
司使（职位） - เสนาบดี
七宿司（机构） - สำนักสัตตะดารา"""

pn_zh2es_source = """原文：这个	译文：Esto
原文：换成款冬花	译文：hay que cambiarlo por tusílago,
原文：伯父都能跟你服软	译文：Incluso puedo pedirte disculpas.
原文：你就不能跟你父亲	译文：¿No puedes disculparte
原文：服个软吗	译文：con tu padre?
原文：皇伯父	译文：Tío.
原文：你要听话	译文：Debes ser obediente.
原文：是	译文：Sí.
原文：长青	译文：Changqing.
原文：奴婢在	译文：Presente.
原文：你觉得晏惜委屈吗	译文：¿Crees que Yanxi ha sido agraviado?
原文：世子对凌王殿下	译文：Me temo que a él le resultará difícil
原文：恐怕难以释怀	译文：perdonar al príncipe Ling.
原文：当年凌王做下了这件事	译文：En aquel entonces, el príncipe Ling hizo eso.
原文：朕不戳破	译文：Si no lo expongo,
原文：他便以为神不知鬼不觉	译文：pensará que nadie se da cuenta.
原文：欠下的账也终归要还的	译文：Las deudas contraídas deben pagarse.
原文：何况	译文：Además,
原文：晏惜是朕养大的	译文：Yanxi fue criado por mí.
原文：如今晏惜受了委屈	译文：Ahora que Yanxi ha sido agraviado,
原文：那朕就替他讨个公道	译文：haré justicia para él.
原文：这次	译文：Esta vez,
原文：父亲没回玉清山躲着	译文：no volviste a esconderte en la montaña Yuqing.
原文：司使忠于官家	译文：Yo, el comandante, soy leal a Su Majestad
原文：七宿卫便该忠于司使	译文：y los guardias del Departamento de Seguridad son leales a mí.
原文：京中之事	译文：Así que el Departamento de Seguridad
原文：七宿司无所不知	译文：debería estar al tanto de todos los asuntos de la capital."""

pn_zh2es_identify = """款冬花（物品） - tusílago
皇伯父（称谓） - Tío
长青（人名） - Changqing
晏惜（人名） - Yanxi
凌王（称谓） - príncipe Ling
玉清山（地名） - montaña Yuqing
司使（职位） - el comandante
七宿司（机构） - Departamento de Seguridad"""

pn_zh2vi_source = """原文：这个	译文：Cái này
原文：换成款冬花	译文：đổi thành khoản đông hoa,
原文：陪你父亲好好说说话	译文：Nói chuyện tử tế với phụ thân của con.
原文：那件事	译文：Chuyện kia
原文：就翻过去吧	译文：hãy để nó sang trang đi.
原文：伯父都能跟你服软	译文：Bá phụ còn có thể nhượng bộ con.
原文：你就不能跟你父亲	译文：Chẳng lẽ con không thể nhượng bộ
原文：服个软吗	译文：phụ thân con sao?
原文：皇伯父	译文：Hoàng bá phụ.
原文：你要听话	译文：Con phải vâng lời.
原文：是	译文：Vâng.
原文：长青	译文：Trường Thanh.
原文：奴婢在	译文：Có nô tài.
原文：你觉得晏惜委屈吗	译文：Ngươi cảm thấy Yến Tích có tủi thân không?
原文：世子对凌王殿下	译文：E là thế tử khó mà buông bỏ được
原文：恐怕难以释怀	译文：chuyện Lăng vương điện hạ.
原文：当年凌王做下了这件事	译文：Năm xưa, Lăng vương làm ra chuyện này
原文：朕不戳破	译文：trẫm không vạch trần.
原文：他便以为神不知鬼不觉	译文：Thế là đệ ấy tưởng thần không biết quỷ không hay.
原文：如今晏惜把旧事捅破了	译文：Nay Yến Tích vạch trần chuyện cũ
原文：也是好事	译文：cũng là cái tốt.
原文：欠下的账也终归要还的	译文：Nợ nào rồi cũng phải trả thôi.
原文：何况	译文：Huống chi,
原文：晏惜是朕养大的	译文：Yến Tích được trẫm nuôi lớn.
原文：如今晏惜受了委屈	译文：Nay Yến Tích chịu thiệt thòi,
原文：那朕就替他讨个公道	译文：trẫm sẽ đòi lại công bằng cho nó.
原文：祖母	译文：Tổ mẫu.
原文：您这一撒手	译文：Người nhắm mắt xuôi tay,
原文：我就再也没有人	译文：con chẳng còn ai
原文：可以依傍了	译文：để dựa dẫm nữa.
原文：逆子	译文：Nghịch tử.
原文：这次	译文：Lần này,
原文：父亲没回玉清山躲着	译文：phụ thân không về núi Ngọc Thanh trốn sao?"""

pn_zh2vi_identify = """款冬花（物品） - khoản đông hoa
皇伯父（称谓） - Hoàng bá phụ
长青（人名） - Trường Thanh
晏惜（人名） - Yến Tích
凌王（称谓） - Lăng vương
朕（称谓） - trẫm
祖母（称谓） - Tổ mẫu
玉清山（地名） - núi Ngọc Thanh"""

pn_zh2ms_source = """原文：这个	译文：Ini,
原文：换成款冬花	译文：tukar kepada terssilage farfara,
原文：陪你父亲好好说说话	译文：Berbual baik-baik dengan ayahanda awak.
原文：那件事	译文：Biarkan hal itu
原文：就翻过去吧	译文：berlalu saja.
原文：伯父都能跟你服软	译文：Pak cik pun boleh bertolak ansur dengan awak,
原文：你就不能跟你父亲	译文：tak boleh awak bertolak ansur
原文：服个软吗	译文：dengan ayahanda awak?
原文：皇伯父	译文：Pak cik.
原文：你要听话	译文：Awak dengarlah nasihat pak cik.
原文：是	译文：Ya.
原文：长青	译文：Changqing.
原文：奴婢在	译文：Ya.
原文：你觉得晏惜委屈吗	译文：Awak rasa Yanxi terkilan tak?
原文：世子对凌王殿下	译文：Takutlah putera susah nak
原文：恐怕难以释怀	译文：maafkan Raja Ling.
原文：当年凌王做下了这件事	译文：Tahun itu, Raja Ling lakukan hal ini,
原文：朕不戳破	译文：beta tak dedahkannya,
原文：他便以为神不知鬼不觉	译文：maka dia rasa tiada sesiapa yang tahu.
原文：如今晏惜把旧事捅破了	译文：Bagus juga Yanxi bocorkan hal lama
原文：也是好事	译文：sekarang.
原文：欠下的账也终归要还的	译文：Dia tetap perlu bayar balik hutang.
原文：何况	译文：Lagipun,
原文：晏惜是朕养大的	译文：beta yang membesarkan Yanxi.
原文：如今晏惜受了委屈	译文：Sekarang Yanxi rasa terkilan,
原文：那朕就替他讨个公道	译文：beta akan tegakkan keadilan untuknya.
原文：祖母	译文：Nenek.
原文：您这一撒手	译文：Selepas nenek meninggal dunia,
原文：我就再也没有人	译文：tiada orang yang saya boleh
原文：可以依傍了	译文：bergantung lagi.
原文：逆子	译文：Anak tak taat.
原文：这次	译文：Kali ini,
原文：父亲没回玉清山躲着	译文：ayahanda tak balik untuk bersembunyi di Gunung Yuqing?"""

pn_zh2ms_identify = """款冬花（物品） - terssilage farfara
皇伯父（称谓） - pak cik
长青（人名） - Changqing
晏惜（人名） - Yanxi
凌王（称谓） - Raja Ling
朕（称谓） - beta
祖母（称谓） - Nenek
玉清山（地名） - Gunung Yuqing"""

pn_zh2pt_source = """原文：这个	译文：Troque
原文：换成款冬花	译文：esse por unha-de-asno,
原文：陪你父亲好好说说话	译文：Vá e fale com seu pai.
原文：那件事	译文：Acho que podemos
原文：就翻过去吧	译文：virar a página.
原文：伯父都能跟你服软	译文：Se eu sou capaz de flexibilizar as coisas com você,
原文：你就不能跟你父亲	译文：você não poderia
原文：服个软吗	译文：flexibilizar com ele?
原文：皇伯父	译文：Tio.
原文：你要听话	译文：Tem de ser obediente.
原文：是	译文：Sim.
原文：长青	译文：Changqing.
原文：奴婢在	译文：Estou aqui.
原文：你觉得晏惜委屈吗	译文：Acha que Yanxi foi injustiçado?
原文：世子对凌王殿下	译文：Acho que Sua Alteza não perdoe
原文：恐怕难以释怀	译文：Príncipe Ling facilmente.
原文：当年凌王做下了这件事	译文：Quando Príncipe Ling fez aquilo,
原文：朕不戳破	译文：se não tivesse sido exposto,
原文：他便以为神不知鬼不觉	译文：ele pensou que ninguém percebe.
原文：如今晏惜把旧事捅破了	译文：Agora Yanxi desenterrou o passado,
原文：也是好事	译文：isso é bom.
原文：欠下的账也终归要还的	译文：Dívidas devem ser pagas.
原文：何况	译文：Além disso,
原文：晏惜是朕养大的	译文：Yanxi foi criado por mim.
原文：如今晏惜受了委屈	译文：Agora que Yanxi foi injustiçado,
原文：那朕就替他讨个公道	译文：tenho de fazer justiça por ele.
原文：祖母	译文：Vovó.
原文：您这一撒手	译文：Você me deixou,
原文：我就再也没有人	译文：não terei mais ninguém
原文：可以依傍了	译文：com quem contar.
原文：逆子	译文：Rebelde.
原文：这次	译文：Dessa vez,
原文：父亲没回玉清山躲着	译文：não se escondeu de novo no Monte Yuqing."""

pn_zh2pt_identify = """款冬花（物品） - unha-de-asno
皇伯父（称谓） - Tio
长青（人名） - Changqing
晏惜（人名） - Yanxi
凌王（称谓） - Príncipe Ling
祖母（称谓） - Vovó
玉清山（地名） - Monte Yuqing"""

pn_zh2id_source = """原文：这个	译文：Ini
原文：换成款冬花	译文：ganti jadi tussilago,
原文：陪你父亲好好说说话	译文：Temani ayahmu bicara baik-baik.
原文：那件事	译文：Peristiwa itu
原文：就翻过去吧	译文：biarkanlah berlalu.
原文：伯父都能跟你服软	译文：Aku bahkan bisa mengalah padamu,
原文：你就不能跟你父亲	译文：apakah kau tidak bisa
原文：服个软吗	译文：mengalah pada ayahmu?
原文：皇伯父	译文：Paman.
原文：你要听话	译文：Kau harus patuh.
原文：是	译文：Baik.
原文：长青	译文：Changqing.
原文：奴婢在	译文：Hamba ada di sini.
原文：你觉得晏惜委屈吗	译文：Apakah kau merasa Yanxi mengalami ketidakadilan?
原文：世子对凌王殿下	译文：Takutnya Putra Raja Ling sulit
原文：恐怕难以释怀	译文：untuk memaafkan Raja Ling.
原文：当年凌王做下了这件事	译文：Waktu itu Raja Ling melakukan hal seperti itu,
原文：朕不戳破	译文：tetapi aku tidak membocorkannya,
原文：他便以为神不知鬼不觉	译文：jadi dia pikir tidak ada yang tahu.
原文：如今晏惜把旧事捅破了	译文：Sekarang Yanxi membocorkan peristiwa lama itu,
原文：也是好事	译文：ini juga termasuk hal yang baik.
原文：欠下的账也终归要还的	译文：Utang pada akhirnya harus dibayar.
原文：何况	译文：Apalagi,
原文：晏惜是朕养大的	译文：Yanxi dibesarkan olehku.
原文：如今晏惜受了委屈	译文：Sekarang Yanxi mengalami perlakuan tidak adil,
原文：那朕就替他讨个公道	译文：jadi aku akan menegakkan keadilan untuk dia.
原文：祖母	译文：Nenek.
原文：您这一撒手	译文：Begitu Nenek melepas tangan,
原文：我就再也没有人	译文：tidak ada orang lagi
原文：可以依傍了	译文：yang bisa kuandalkan.
原文：逆子	译文：Anak durhaka.
原文：这次	译文：Kali ini,
原文：父亲没回玉清山躲着	译文：Ayah tidak kembali ke Gunung Yuqing untuk bersembunyi?"""

pn_zh2id_identify = """款冬花（物品） - tussilago
皇伯父（称谓） - Paman
长青（人名） - Changqing
晏惜（人名） - Yanxi
凌王（称谓） - Raja Ling
朕（称谓） - aku
祖母（称谓） - Nenek
玉清山（地名） - Gunung Yuqing"""

pn_en2zh_source = """原文：Who are you?	译文：你是谁
原文：I am sheldon's cousin leo.	译文：我是谢尔顿的表弟里奥
原文：Oh,God.	译文：天啊
原文：Sheldon does not have a cousin leo.	译文：谢尔顿根本没有一个叫里奥的表弟
原文：Au contraire.	译文：恰恰相反
原文：I'm 26 years old.	译文：我今年26岁
原文：I'm originally from... denton, texas,	译文：我的祖籍是 德克萨斯州的登顿市
原文：But I was a navy brat, so I was	译文：但我是个海军顽童  所以我从小
原文：Brought up on a variety of military bases around the world.	译文：是在世界各地的军事基地长大的
原文：As a result, I've often felt like an outsider--	译文：因此  我总感觉跟群体格格不入
原文：Never really fitting in,	译文：从没真正融入过
原文：Which is probably the reason for my substance abuse problem.	译文：这也很可能是我滥用毒品的原因
原文：Excuse me, we just went over this.	译文：打扰一下  我们刚才还练习过一遍
原文：As the quintessential middle child, your addiction	译文：作为一个典型的中间儿  你的毒瘾
原文：Is rooted in your unmet need for attention.	译文：根植于你未被满足的对关注度的需求当中
原文：Oh, sheldon, are we really going to go with pop psychology?	译文：谢尔顿  我们不是真的要用通俗心理学吧
原文：For your information, this is all based on solid research.	译文：请注意  这些都是以有力的研究为根据的
原文：Just stick with the character profile I wrote for you.	译文：完全依照我写给你的性格档案就行了
原文：Sheldon? I'm sorry.	译文：-谢尔顿  -不好意思
原文：This is toby loobenfeld.	译文：莱纳德  这位是托比·鲁本菲尔德
原文：He's a research assistant	译文：他是粒子物理实验室的
原文：In the particle physics lab,	译文：一名研究助理
原文：But he also minored in theater at mit.	译文：不过他同时在麻省理工辅修戏剧
原文：It was more of a double major, actually.	译文：事实上  说双学位更准确些"""

pn_en2zh_identify = """sheldon（人名） - 谢尔顿
leo（人名） - 里奥
denton（地名） - 登顿
texas（地名） - 德克萨斯州
navy brat（称谓） - 海军顽童
military bases（机构） - 军事基地
character profile（物品） - 性格档案
particle physics lab（机构） - 粒子物理实验室
double major（其他） - 双学位"""

pn_ja2zh_source = """原文：男は マグロ漁船で５年	译文：男孩子们可以 去捕金枪鱼船上工作五年
原文：女は吉原よしわらで３年も働きゃ 返せんだろ	译文：女孩子去吉原红灯区工作三年 很快就能还清了
原文：俺をリングに上げてください	译文：请让我上擂台
原文：龍になるのがハッタリじゃないって 分からせてみせます	译文：我会向您证明我能成为“人中之龙”
原文：黙れ 一馬！	译文：住口 一马！
原文：俺が勝ったら組に入れてください	译文：如果我赢了 请让我加入组织
原文：カネは働いて返します	译文：我会将功赎罪的
原文：負けたら どうすんだ？　あ？	译文：如果你输了呢？那怎么办？
原文：そんときは…	译文：那…
原文：好きにしてください 一馬！	译文：您想怎么处置我都行 一马！
原文：お前 何言ってんだよ	译文：你在说什么？
原文：菅	译文：克己
原文：あいつに誰か対戦相手 手配できるか？	译文：你能给他找个对手吗？
原文：いつ… ですか？	译文：什么时候？
原文：そりゃあ お前 今晩に決まってんだろ	译文：当然是今晚
原文：ありがとうございます！	译文：谢谢您 先生
原文：なんで 黙ってたんだよ お前 なんで 一人で決めちまうんだよ！	译文：你为什么不和我们商量 就擅自决定一切？
原文：やめてお兄ちゃん	译文：别说了 阿錦！ 我们不是一家人吗？
原文：お前 家族じゃなかったのかよ！ 錦！	译文：别说了 阿錦！ 我们不是一家人吗？
原文：訳 分かんねえよ もう 風間がヤクザとかよ！	译文：简直瞎搞 你为什么要加入黑道？
原文：堂島組のカネに手出してよ…	译文：我们偷了堂岛组的钱！ 我们该怎么办？
原文：俺は試合に勝つから	译文：我一定会打赢的 搞什么…
原文：勝って堂島の龍になる	译文：我会打赢 然后成为堂岛之龙
原文：なって どうする？	译文：然后呢？
原文：このクソみてえな世界を 変えるんだよ	译文：我要改变这个糟糕的世界
原文：お前は何も分かってない	译文：你根本什么都不懂
原文：極道のことも 龍のことも	译文：无论是黑道 还是“人中之龙”
原文：俺は お前たちを遠ざけたかった	译文：我一直都想让你们…
原文：こういう…	译文：远离…
原文：世界から	译文：这个世界"""

pn_ja2zh_identify = """マグロ漁船（机构） - 捕金枪鱼船
吉原よしわら（地名） - 吉原红灯区
リング（物品） - 擂台
一馬（人名） - 一马
堂島（人名） - 堂岛
極道（称谓） - 黑道
龍（称号） - 人中之龙
風間（人名） - 风间
錦（人名） - 阿锦"""

pn_ko2zh_source = """原文：자네가 이해하고	译文：你就谅解一下
原文：뭐 아는 게 있으면 얘기 좀 해 주게	译文：若你知道些什么 就告诉我吧
原文：내 사례는 넉넉히 하겠네	译文：我会好好酬谢你的
原文：몇 번을 말씀드립니까	译文：你要我说几次？
原文：전 의원님을 못 뵌 지 몇 달이 넘었습니다	译文：我已经好几个月没见到医员了
原文：작은 거라도 좋으니 얘기 좀 해 주게	译文：即使小事也好 你就告诉我吧
原文：너무 중요해서 그러네	译文：因为我们有非常重要的事
原文：내 관에는 절대로 알리지 않겠네	译文：我绝不会让官衙的人知道
原文：어제	译文：昨天
原文：지율헌에서 일하던 서비라는 의녀가 찾아왔었습니다	译文：有位在持律轩工作 名叫舒菲的医女来找过我
原文：지율헌 사건에 생존자가 있단 말이냐?	译文：持律轩事件有幸存者吗？
原文：지금 그 의녀는 어디 있는 것이냐	译文：那位医女现在在哪里？
原文：잘은 모르겠지만	译文：我不太清楚
原文：아마 언골로 갔을 겁니다	译文：但她可能去冻谷了
原文：절 찾아와선	译文：她来找我
原文：언골에서 난다는 생사초라는 풀에 대해 이것저것 물었거든요	译文：针对生长在那里的生死草 问了许多问题
原文：죽은 사람을 살리는 풀이니 뭐니	译文：听说那种药草能救活死人
原文：혼이 빠진 게	译文：她整个人失魂落魄的
原文：보통 사람으로 보이진 않았습니다	译文：看起来不像正常人
原文：방금 뭐라 하였느냐	译文：你刚才说什么？
原文：죽은 사람을 살리는 풀?	译文：能救活死人的药草？
原文：예	译文：是的
原文：하나 세상천지 그런 풀이 어디 있겠습니까	译文：但天底下怎么可能有那种药草？
原文：그 언골이라는 곳이 어디냐	译文：那个名叫冻谷的地方在哪里？
原文：사시사철 얼음이 어는 계곡이라 하여	译文：那个溪谷一年四季都结冰
原文：언골이라 부르는데	译文：因此称作冻谷
原文：동래 북쪽 고미산 속에 있다 들었습니다	译文：我听说它位于东莱北方的高弥山里
原文：저하께선 여기 계십시오 제가 다녀오겠습니다	译文：邸下 请您待在这里 我去就好
原文：아니다 혼자 있는 게 더 무섭다	译文：不 独自待着更可怕
原文：이곳은 벌써 한겨울이구나	译文：这里已经是冬天了啊
原文：저하 제 뒤에 계십시오	译文：邸下 请您站在我身后
原文：저하	译文：邸下
原文：저하 위험합니다	译文：邸下 很危险
原文：지율헌에서 죽은 의녀들과 같은 옷이다	译文：那身衣服和持律轩死去的医女一样
原文：네가 지율헌 서비라는 의녀냐?	译文：你是持律轩里叫舒菲的医女吗？"""

pn_ko2zh_identify = """지율헌（机构） - 持律轩
서비（人名） - 舒菲
언골（地名） - 冻谷
생사초（物品） - 生死草
동래（地名） - 东莱
고미산（地名） - 高弥山
저하（称谓） - 邸下"""

pn_en2de_source = """原文：At least you kept your eyes.	译文：Wenigstens hast du deine Augen behalten.
原文：His loss.	译文：Sein Pech.
原文：Piss off.	译文：Verpiss dich.
原文：Well, I would, but we don't have much time.	译文：Das würde ich, aber die Zeit ist knapp.
原文：Who the fuck are you?	译文：Wer bist du?
原文：I'm Vilgefortz of Roggeveen.	译文：Ich bin Vilgefortz von Roggeveen.
原文：Whatever you lack in talent, you make up for in confidence.	译文：Was dir an Talent fehlt, machst du mit Selbstvertrauen wett.
原文：She doesn't need confidence. Her father owns half of Creyden.	译文：Das braucht sie nicht. Ihrem Vater gehört halb Creyden.
原文：So he could swap a hundred horses for her spot here.	译文：Er hat 100 Pferde für ihren Platz eingetauscht.
原文：Your parents paid Aretuza?	译文：Deine Eltern haben Aretusa bezahlt?
原文：The Chapter decided it needed students from the best families.	译文：Das Kapitel brauchte Schüler aus den besten Familien.
原文：But you all must have had a conduit moment?	译文：Aber ihr müsst einen Mittler-Moment gehabt haben?
原文：We shouldn't be mixing herbs. We shouldn't even be here.	译文：Wir sollen keine Kräuter mischen. Wir sollten nicht mal hier sein.
原文：You know, if they catch us, they really will expel us.	译文：Wenn wir erwischt werden, werfen sie uns raus.
原文：There are far worse things than expulsion.	译文：Es gibt weitaus Schlimmeres.
原文：Like what?	译文：Was zum Beispiel?
原文：What is this place?	译文：Was ist das für ein Ort?
原文：Aretuza's windmill.	译文：Aretusas Windmühle.
原文：Enough bridled chaos to keep the curtains hung...	译文：Genug gezügeltes Chaos, um die Vorhänge aufgehängt und die Fackeln
原文：and the torches lit, but that's not what we're here for.	译文：angezündet zu halten, aber dafür sind wir nicht da."""

pn_en2de_identify = """Vilgefortz（人名） - Vilgefortz
Roggeveen（地名） - Roggeveen
Creyden（地名） - Creyden
Aretuza（地名） - Aretusa
The Chapter（机构） - Das Kapitel"""

pn_en2fr_source = """原文：That dreadful play my wife forced me to attend.	译文：Cette farce grotesque que ma femme m'a forcé à regarder.
原文：I've never forgiven her.	译文：Je ne lui ai jamais pardonné.
原文：Colonel Fraser recently returned from Scotland,	译文：Le colonel Fraser est récemment revenu d'Écosse
原文：and brought us some correspondence.	译文：et nous rapporte de la correspondance.
原文：I had occasion to travel to France,	译文：J'ai dû me rendre en France,
原文：thought you would be glad to receive word	译文：j'ai pensé vous donner des nouvelles
原文：of the generous contributions to our cause.	译文：des généreuses contributions à notre cause.
原文：Remarkable.	译文：Remarquable.
原文：-You did this of your own accord? -Aye.	译文：- Vous avez agi de votre propre chef ? - Oui.
原文：Sit with me, Colonel Fraser.	译文：Venez vous asseoir avec moi.
原文：You're aware that Clinton is preparing to withdraw from Philadelphia?	译文：Vous savez que Clinton s'apprête à se retirer de Philadelphie ?
原文：I heard an evacuation is already in progress.	译文：J'ai entendu dire qu'une évacuation était déjà en cours.
原文：I'm impressed with your cunning in securing these documents.	译文：Je suis impressionné par votre ruse pour sécuriser ces documents.
原文：You took a Loyalist's favor and turned it into a boon for us.	译文：Transformer un service en une telle aubaine.
原文：And Colonel Morgan has extolled your bravery on the battlefield at Saratoga.	译文：Et le colonel Morgan a vanté votre bravoure à la bataille de Saratoga.
原文：Will you do me the honor of accepting command of a battalion?	译文：Me ferez-vous l'honneur d'accepter le commandement d'un bataillon ?
原文：I, uh…	译文：Je…
原文：I'd be exceedingly honored, sir.	译文：J'en serais extrêmement honoré, monsieur.
原文：Very well, then.	译文：Très bien.
原文：You're appointed Brigadier General.	译文：Vous êtes nommé général de brigade.
原文：Thank you, sir.	译文：Merci, monsieur.
原文：Although, the Congress will have to approve your appointment.	译文：Cependant, le Congrès devra approuver votre nomination."""

pn_en2fr_identify = """Colonel（称谓） - colonel
Fraser（人名） - Fraser
Scotland（地名） - Écosse
France（地名） - France
Clinton（人名） - Clinton
Philadelphia（地名） - Philadelphie
Morgan（人名） - Morgan
Saratoga（地名） - Saratoga
Brigadier General（职位） - général de brigade
Congress（机构） - Congrès"""

pn_en2th_source = """原文：Look, Suthon is approaching.	译文：ดูสิ สุทนกำลังมา
原文：We should pay respect.	译文：เราควรแสดงความเคารพ
原文：Khun Chai, your father is waiting.	译文：คุณชาย พ่อท่านกำลังรอ
原文：The shipment from Thalang Bay arrived.	译文：สินค้าจากอ่าวถลางมาถึงแล้ว
原文：This box contains Emerald Lotus.	译文：กล่องนี้มีดอกบัวมรกต
原文：The Siamese Guild requests your presence.	译文：สยามกิลด์ขอให้ท่านไปร่วม
原文：Prepare the Golden Naga statue.	译文：เตรียมรูปปั้นนาคทอง
原文：The ceremony at Wat Arun starts at dusk.	译文：พิธีที่วัดอรุณเริ่มตอนพลบค่ำ
原文：Where is Master Arun?	译文：อาจารย์อรุณอยู่ที่ไหน
原文：Check the eastern pavilion.	译文：ดูที่ศาลาทิศตะวันออก
原文：The royal decree mentions Ayutthaya Palace.	译文：พระราชโองการกล่าวถึงวังอยุธยา
原文：Protect the Nine-Gem Ring at all costs.	译文：ต้องปกป้องแหวนเก้ายมกให้ได้
原文：The rebels are near Sukhothai Gate.	译文：กบฏอยู่ใกล้ประตูสุโขทัย
原文：Khun Ying, the tea is ready.	译文：คุณหญิง น้ำชาพร้อมแล้ว
原文：The monk from Nakhon Pathom left this.	译文：พระจากนครปฐมทิ้งสิ่งนี้ไว้
原文：We need more silk from Ban Krua.	译文：ต้องการผ้าไหมจากบ้านครัวเพิ่ม
原文：The astrologer predicted heavy rain.	译文：โหรทำนายว่าฝนจะตกหนัก
原文：Where's the Golden Naga?	译文：นาคทองอยู่ไหน
原文：It's enshrined in the main hall.	译文：ประดิษฐานอยู่ในหอหลัก
原文：The Siamese Guild awaits your decision.	译文：สยามกิลด์รอคำตัดสินของท่าน
原文：The Emerald Lotus must bloom tonight.	译文：ดอกบัวมรกตต้องบานคืนนี้
原文：Khun Chai, the guards are restless.	译文：คุณชาย ยามเริ่มกระสับกระส่าย
原文：Check the secret passage near Sukhothai Gate.	译文：สำรวจทางลับใกล้ประตูสุโขทัย
原文：The Nine-Gem Ring shines brighter.	译文：แหวนเก้ายมกสว่างขึ้น
原文：Master Arun is meditating by the river.	译文：อาจารย์อรุณกำลังนั่งสมาธิริมน้ำ
原文：Prepare the boat for Thalang Bay.	译文：เตรียมเรือไปอ่าวถลาง"""

pn_en2th_identify = """Suthon（人名） - สุทน
Khun Chai（称谓） - คุณชาย
Thalang Bay（地名） - อ่าวถลาง
Emerald Lotus（物品） - ดอกบัวมรกต
Siamese Guild（机构） - สยามกิลด์
Golden Naga（物品） - นาคทอง
Wat Arun（地名） - วัดอรุณ
Nine-Gem Ring（物品） - แหวนเก้ายมก
Sukhothai Gate（地名） - ประตูสุโขทัย
Khun Ying（称谓） - คุณหญิง
Nakhon Pathom（地名） - นครปฐม
Ban Krua（地名） - บ้านครัว
Master Arun（人名） - อาจารย์อรุณ
Ayutthaya Palace（地名） - วังอยุธยา"""

pn_en2vi_source = """原文：We need to find the Shadowstone in Ravenspire.	译文：Chúng ta cần tìm Shadowstone ở Ravenspire.
原文：Commander Adler will meet us at Frostfang Keep.	译文：Chỉ huy Adler sẽ gặp chúng ta tại Pháo đài Frostfang.
原文：The Ironhand Guild controls the northern mines.	译文：Công hội Ironhand kiểm soát các mỏ phía bắc.
原文：Elena found ancient texts in Blackmarsh Swamp.	译文：Elena tìm thấy văn tự cổ trong Đầm Blackmarsh.
原文：This map shows the Temple of Dusk's location.	译文：Bản đồ này chỉ vị trí Đền Dusk.
原文：High Priestess Mira forbids entering the catacombs.	译文：Nữ Giáo chủ Mira cấm vào hầm mộ.
原文：The Obsidian Blade was last seen in Silverhaven.	译文：Lưỡi kiếm Obsidian được nhìn thấy cuối ở Silverhaven.
原文：Prepare the wyverns for dawn departure.	译文：Chuẩn bị wyverns cho chuyến đi lúc bình minh.
原文：The Shadow Emperor's forces are mobilizing.	译文：Lực lượng của Shadow Emperor đang di chuyển.
原文：Elena, decode these runes from Frostfang.	译文：Elena, giải mã các ký tự từ Frostfang đi.
原文：The Guild demands 20% of ore profits.	译文：Công hội yêu cầu 20% lợi nhuận quặng.
原文：Silverhaven's council denied our request.	译文：Hội đồng Silverhaven từ chối yêu cầu của ta.
原文：Mira warned about the Temple's curse.	译文：Mira cảnh báo về lời nguyền của ngôi đền.
原文：Adler wants reinforcements at Blackmarsh.	译文：Adler muốn viện binh tới Blackmarsh.
原文：The Blade can only be wielded by Dusk's chosen.	译文：Lưỡi kiếm chỉ dành cho người được Dusk chọn.
原文：Wyverns need special saddles for long flights.	译文：Wyverns cần yên đặc biệt cho hành trình dài.
原文：The Emperor's spies infiltrated Ravenspire.	译文：Gián điệp của Emperor đã xâm nhập Ravenspire.
原文：These mines connect to the Temple's basement.	译文：Các mỏ này thông với tầng hầm đền.
原文：Dusk's prophecy mentions twin moons alignment.	译文：Lời tiên tri Dusk nhắc đến sự xếp hàng của song nguyệt.
原文：Mira will perform the purification ritual.	译文：Mira sẽ thực hiện nghi thức thanh tẩy.
原文：The Guild's emblem bears a hammer and anvil.	译文：Biểu tượng Công hội có búa và đe.
原文：Adler requests more scouts for Silverhaven.	译文：Adler yêu cầu thêm trinh sát tới Silverhaven.
原文：Elena discovered hidden chambers in Frostfang.	译文：Elena phát hiện phòng ẩn trong Frostfang.
原文：Wyvern riders must report to Adler before dusk.	译文：Kỵ sĩ wyvern phải báo cáo với Adler trước hoàng hôn.
原文：The Shadowstone's power could awaken the Emperor.	译文：Sức mạnh Shadowstone có thể đánh thức Emperor."""

pn_en2vi_identify = """Shadowstone（物品） - Shadowstone
Ravenspire（地名） - Ravenspire
Commander Adler（称谓） - Chỉ huy Adler
Frostfang（地名） - Frostfang
Ironhand Guild（机构） - Công hội Ironhand
Elena（人名） - Elena
Blackmarsh（地名） - Blackmarsh
High Priestess Mira（称谓） - Nữ Giáo chủ Mira
Temple of Dusk（机构） - Đền Dusk
Obsidian Blade（物品） - Lưỡi kiếm Obsidian
Silverhaven（地名） - Silverhaven
wyverns（物品） - wyverns
Shadow Emperor（称谓） - Shadow Emperor"""

proper_noun_slot_dict = {
    "zh2en": ["中文", "英文", "中文", "英文", pn_zh2en_source, pn_zh2en_identify],
    "zh2th": ["中文", "泰国语", "中文", "泰国语", pn_zh2th_source, pn_zh2th_identify],
    "zh2es": ["中文", "西班牙语", "中文", "西班牙语", pn_zh2es_source, pn_zh2es_identify],
    "zh2vi": ["中文", "越南语", "中文", "越南语", pn_zh2vi_source, pn_zh2vi_identify],
    "zh2ms": ["中文", "马来语", "中文", "马来语", pn_zh2ms_source, pn_zh2ms_identify],
    "zh2pt": ["中文", "葡萄牙语", "中文", "葡萄牙语", pn_zh2pt_source, pn_zh2pt_identify],
    "zh2id": ["中文", "印尼语", "中文", "印尼语", pn_zh2id_source, pn_zh2id_identify],
    "en2zh": ["英文", "中文", "英文", "中文", pn_en2zh_source, pn_en2zh_identify],
    "ja2zh": ["日语", "中文", "日语", "中文", pn_ja2zh_source, pn_ja2zh_identify],
    "ko2zh": ["韩语", "中文", "韩语", "中文", pn_ko2zh_source, pn_ko2zh_identify],
    "en2de": ["英文", "德语", "英文", "德语", pn_en2de_source, pn_en2de_identify],
    "en2fr": ["英文", "法语", "英文", "法语", pn_en2fr_source, pn_en2fr_identify],
    "en2th": ["英文", "泰国语", "英文", "泰国语", pn_en2th_source, pn_en2th_identify],
    "en2vi": ["英文", "越南语", "英文", "越南语", pn_en2vi_source, pn_en2vi_identify],
}

proper_noun_pe_template = """【{}到{}术语及其译文识别】

<要求>
在进行翻译时，某些专有名词是指那些独特且专指某一人名、地名、机构、物品或事件等的名称。翻译专有名词时，需要格外注意不同文化和语言环境下的使用习惯，确保专有名词的准确性尤为重要。现在请你帮助我识别出{}影视剧台词中的专有名词，要求如下：
1. 识别剧中出现的需要专门翻译的名词；
2. 标识出识别到的专有名词的类型，类型仅限于人名、称谓、地名、机构、物品、职位、事件、其他这八个；
3. 如果没有识别到专有名词直接输出“无专有名词”即可，对于一些常用词无需作为专有名词。
4. 如果识别到存在专有名词请直接输出识别结果，并根据译文输出相应专有名词的{}翻译，输出格式参考以下样例。

<样例>
原台词及其译文：
{}

识别到的专有名词：
{}

<任务>
现在按照前面提到的要求，并参考上面提供的样例，从下面的台词中识别出专有名词及其翻译。
原台词及其译文：
{}

请按照样例的格式识别上面的原文中的专有名词。注意只需要按照格式输出专有名词的识别结果及其译文即可。"""

# 与上面第二个要求不一样
proper_noun_train_template = """【{}到{}术语识别与翻译】

<要求>
在进行翻译时，某些专有名词是指那些独特且专指某一人名、地名、机构、物品或事件等的名称。翻译专有名词时，需要格外注意不同文化和语言环境下的使用习惯，确保专有名词的准确性尤为重要。现在请你帮助我识别出{}影视剧台词中的专有名词，要求如下：
1. 识别剧中出现的需要专门翻译的名词；
2. 标识出识别到的专有名词的类型，类型包括人名、称谓、地名、机构、物品、职位、事件、其他等；
3. 如果没有识别到专有名词直接输出“无专有名词”即可，对于一些常用词无需作为专有名词。
4. 如果识别到存在专有名词请直接输出识别结果，并根据译文输出相应专有名词的{}翻译，输出格式参考以下样例。

<样例>
{}

识别到的专有名词：
{}

<任务>
现在按照前面提到的要求，并参考上面提供的样例，从下面的台词中识别出专有名词及其翻译。
{}

请按照样例的格式识别上面的原文中的专有名词。注意只需要按照格式输出专有名词的识别结果及其译文即可。
"""

translation_template = """【<src_lang_str>到<lang_str>台词翻译】

<<1>>
1.<<2>>
2.<<3>>
3.<<4>>
4.<<5>>

专有名词翻译：
{}

<<6>>
原文：
{}

翻译结果：
"""

yingshi_choice = ['影视', '影视剧', '剧目', '影视剧目', '节目', '影视节目']
yaoqiu_choice = ['以下要求', '如下要求', '下列要求', '下面的要求']
taici_choice = ['台词', '字幕', '对白']
yiwen_choice = ['译文', '翻译']
tongsuyidong_choice = ['通俗易懂', '浅显易懂', '易于理解', '容易理解']
yuyanfengge_choice = ['语言风格', '表达风格']
yizhi_choice = ['一致', '保持一致', '相匹配', '相一致']
xuyao_choice = ['需要', '要', '必须', '应该']
yitong_choice = ['一同', '一并', '一起']
anzhao_choice = ['按照', '遵照', '遵循']

def get_replace_dict():
    taici = random.choice(taici_choice)
    replace_dict = {
        "<<1>>": [
            f"请将多条<src_lang_str>{random.choice(yingshi_choice)}{taici}翻译成<lang_str>，{random.choice(anzhao_choice)}{random.choice(yaoqiu_choice)}：",
            f"请{random.choice(anzhao_choice)}{random.choice(yaoqiu_choice)}来将多条<src_lang_str>{random.choice(yingshi_choice)}{taici}翻译成<lang_str>：",
            f"翻译以下多条<src_lang_str>{random.choice(yingshi_choice)}{taici}为<lang_str>，{random.choice(anzhao_choice)}{random.choice(yaoqiu_choice)}：",
            f"请{random.choice(anzhao_choice)}{random.choice(yaoqiu_choice)}，将下面的多句<src_lang_str>{random.choice(yingshi_choice)}{taici}译成<lang_str>：",
            f"请{random.choice(anzhao_choice)}{random.choice(yaoqiu_choice)}，把一系列<src_lang_str>{random.choice(yingshi_choice)}{taici}翻译成<lang_str>："
        ],
        "<<2>>": [
            f"{random.choice(yiwen_choice)}{random.choice(xuyao_choice)}口语化，{random.choice(tongsuyidong_choice)}，与<src_lang_str>{taici}{random.choice(yuyanfengge_choice)}{random.choice(yizhi_choice)}。",
            f"{random.choice(yiwen_choice)}{random.choice(xuyao_choice)}与<src_lang_str>{taici}{random.choice(yuyanfengge_choice)}{random.choice(yizhi_choice)}，口语化且{random.choice(tongsuyidong_choice)}。",
            f"确保{random.choice(yiwen_choice)}的语言是口语化的、{random.choice(tongsuyidong_choice)}的，并且{random.choice(yuyanfengge_choice)}{random.choice(xuyao_choice)}与<src_lang_str>的{taici}{random.choice(yizhi_choice)}。"
        ],
        "<<3>>": [
            f"翻译的<lang_str>{taici}{random.choice(xuyao_choice)}与<src_lang_str>长度{random.choice(yizhi_choice)}。",
            f"每句{random.choice(yiwen_choice)}的长度{random.choice(xuyao_choice)}与<src_lang_str>{random.choice(yizhi_choice)}",
            f"<lang_str>{random.choice(yiwen_choice)}{random.choice(xuyao_choice)}与<src_lang_str>原文的长度{random.choice(yizhi_choice)}。",
            f"确保译成<lang_str>的{taici}与其<src_lang_str>原{taici}的长度{random.choice(yizhi_choice)}。",
            f"<lang_str>{random.choice(yiwen_choice)}{random.choice(xuyao_choice)}保持与<src_lang_str>原文相同的长度。",
            f"翻译成<lang_str>的{taici}长度{random.choice(xuyao_choice)}和<src_lang_str>版{random.choice(yizhi_choice)}。"
        ],
        "<<4>>": [
            f"专有名词{random.choice(xuyao_choice)}翻译成指定的译文。",
            f"{random.choice(xuyao_choice)}将专有名词译为规定的对应译文。",
            f"{random.choice(xuyao_choice)}确保将专有名词翻译为指明的译文。",
            f"翻译专有名词时，{random.choice(xuyao_choice)}使用指定的译文。",
            f"{random.choice(xuyao_choice)}将专有名词按照指定方式进行翻译。",
            f"专有名词{random.choice(xuyao_choice)}按照所指定的翻译来准确转译。"
        ],
        "<<5>>": [
            f"确保输出与提供的原文行数{random.choice(yizhi_choice)}，不要合并{taici}，{random.choice(yitong_choice)}输出原文与译文。",
            f"请确保输出{random.choice(yiwen_choice)}与所给原文的行数{random.choice(yizhi_choice)}，避免合并{taici}内容，将原文与译文{random.choice(yitong_choice)}输出。",
            f"输出时{random.choice(xuyao_choice)}保证{random.choice(yiwen_choice)}行数与原始文本{random.choice(yizhi_choice)}，避免将{taici}融合，原文和翻译{random.choice(xuyao_choice)}{random.choice(yitong_choice)}输出。",
            f"不要合并{taici}，将原文与{random.choice(yiwen_choice)}{random.choice(yitong_choice)}输出，输出时{random.choice(yiwen_choice)}行数与<src_lang_str>原文{random.choice(yizhi_choice)}。",
            f"将原文与{random.choice(yiwen_choice)}{random.choice(yitong_choice)}输出，注意不能合并{taici}，在输出时{random.choice(yiwen_choice)}行数与<src_lang_str>原文{random.choice(xuyao_choice)}{random.choice(yizhi_choice)}。"
        ],
        "<<6>>": [
            f"现在请{random.choice(anzhao_choice)}要求，完成以下{taici}翻译。",
            f"{random.choice(anzhao_choice)}上面的要求，翻译以下{taici}。",
            f"请{random.choice(anzhao_choice)}前面的要求，翻译下面的{taici}。",
            f"{random.choice(anzhao_choice)}前面所提出的要求，完成下面的{taici}翻译。"
        ]
    }
    return replace_dict

term_adjust_few_shot = """【例子1：英译中】
1.
原文：Mr. Woo Yeon-woo
译文：佑延佑先生
类型：称谓
2.
原文：Woo Yeon-woo
译文：禹延佑
类型：人名
3.
原文：Jong-woo
译文：钟佑
类型：人名
4.
原文：Yeon-woo
译文：廷祐
类型：人名

术语翻译调整结果：
{
    "Mr. Woo Yeon-woo": {"require correction": true, "translation": "佑延佑先生"},
    "Woo Yeon-woo": {"require correction": false, "translation": "禹延佑"},
    "Jong-woo": {"require correction": false, "translation": "钟佑"},
    "Yeon-woo": {"require correction": true, "translation": "延佑"}
}

【例子2：中译英】
1.
原文：阿福
译文：Fu
类型：人名
2.
原文：傻阿福
译文：Silly AFU
类型：称谓
3.
原文：阿福媳妇
译文：Ahfu's wife
类型：称谓
4.
原文：阿福公子
译文：Mr. Fu
类型：称谓
5.
原文：阿福大夫
译文：Dr. Futu
类型：称谓

术语翻译调整结果：
{
    "阿福": {"require correction": false, "translation": "Fu"},
    "傻阿福": {"require correction": true, "translation": "Silly Fu"},
    "阿福媳妇": {"require correction": true, "translation": "Fu's wife"},
    "阿福公子": {"require correction": false, "translation": "Mr. Fu"},
    "阿福大夫": {"require correction": true, "translation": "Dr. Fu"}
}

【例子3：中译韩】
1.
原文：沈家
译文：심 집안
类型：机构
2.
原文：聂家
译文：녜가문
类型：机构
3.
原文：李家
译文：이 집안
类型：机构
4.
原文：沈家诊所
译文：심가문 의원
类型：机构
5.
原文：聂家少帅
译文：녜가문의 젊은 장군
类型：称谓

术语翻译调整结果：
{
    "沈家": {"require correction": true, "translation": "심가문"},
    "聂家": {"require correction": false, "translation": "녜가문"},
    "李家": {"require correction": true, "translation": "이가문"},
    "沈家诊所": {"require correction": false, "translation": "심가문 의원"},
    "聂家少帅": {"require correction": false, "translation": "녜가문의 젊은 장군"}
}

【例子4：日译中】
1.
原文：田中太郎
译文：田中太郎
类型：人名
2.
原文：田中太郎氏
译文：田中先生
类型：称谓
3.
原文：田中クリニック
译文：田中诊所
类型：机构
4.
原文：田中さん
译文：田中桑
类型：称谓
5.
原文：田中博士
译文：Tanaka博士
类型：称谓

术语翻译调整结果：
{
    "田中太郎": {"require correction": false, "translation": "田中太郎"},
    "田中太郎氏": {"require correction": true, "translation": "田中太郎先生"},
    "田中クリニック": {"require correction": false, "translation": "田中诊所"},
    "田中さん": {"require correction": true, "translation": "田中先生"},
    "田中博士": {"require correction": true, "translation": "田中博士"}
}

【例子5：中译英】
1.
原文：张老师
译文：Mr. Zhang
类型：称谓
2.
原文：赵老师
译文：Teacher Zhao
类型：称谓
3.
原文：陈老师
译文：Teacher Chen
类型：称谓

术语翻译调整结果：
{
    "张老师": {"require correction": false, "translation": "Mr. Zhang"},
    "赵老师": {"require correction": true, "translation": "Mr. Zhao"},
    "陈老师": {"require correction": true, "translation": "Mr. Chen"}
}"""

term_adjust_template = """{}

在影视节目的字幕翻译中保持节目中的术语或者专有名词的翻译一致性是至关重要的，以上是一些为了保证术语翻译一致性而对一些相近词进行术语译文调整的例子。请你仿照上面的例子，完成下面的{}术语译文一致性调整，要求保证：
1.称谓结构一致：当原文包含称谓格式（如英语中的Mr./Dr./头衔）时，需确保姓氏和名字的翻译与单独人名的翻译保持逻辑一致（如例子1中"Woo"在称谓和人名中的汉字选择需统一）；
2.复合术语的合理拆分处理：当复合称谓包含人名+身份（如"阿福媳妇"）时，需保持核心人名的翻译完整性和一致性，避免随意缩写（如从Ahfu's wife调整为Fu's wife）；
3.机构名称的统一规范：涉及家族、种族、机构的翻译需统一用词规范（如韩语例子中"가문"作为家族的标准译法优于直译的"집안"）；
4.文化适配性：需符合目标语言的称谓习惯（如中文"公子"对应"Mr."而非直译；韩语中家族称谓的正式化表达）；
5.音译标准化：同一人名的音译用字需前后统一（如例子2中"阿福"统一为"Fu"而非混合使用AFU/Futu）；

现在请你综合考虑以下提供的多个{}术语与其{}译文的情况，对其中的翻译不一致的情况进行调整，用与例子中一致的json格式返回术语翻译调整结果。如果你认为提供的译文翻译情况合理，则可不必进行调整，输出结果中require correction设置为true即可。以下为提供的{}术语与其{}译文：
{}

术语翻译调整结果（仅输出json格式结果，除此之外不要输出任何其他的内容）：
"""

sentence_segment_template_zh = """【例1】
[不连贯] 娘
[连贯] 爹什么时候回来
[不连贯] 这都是些什么字啊
[不连贯] 你要是感兴趣的话
[连贯] 待出塔以后我教你识字
[不连贯] 真的
[连贯] 可你总是那么忙
[不连贯] 每天抽出一两个时辰总是可以的
[不连贯] 那我怎么知道你是不是在糊弄我
[不连贯] 好
[连贯] 那每天你要陪我玩一个时辰
[不连贯] 对了
[连贯] 还要陪我娘
[连贯] 两个时辰
[不连贯] 好
[连贯] 大丈夫一言既出
[连贯] 驷马难追
[不连贯] 那你可不能说话不算话啊
[不连贯] 不能不算话
[不连贯] 都说了
[连贯] 大丈夫 一言既出 驷马难追
[不连贯] 吒儿
[连贯] 你受伤了吗
[不连贯] 我没事
[连贯] 就是手脚有点使不上劲
[不连贯] 我们去前面休息一下

【例2】
[不连贯] 虽然有些回忆不太美好
[连贯] 但都算咱们俩
[连贯] 最珍贵的回忆了吧
[不连贯] 幸好
[连贯] 你现在还在我身边
[不连贯] 以前你是我的光
[连贯] 以后啊
[连贯] 我要做你的光
[连贯] 我要陪在你身边
[不连贯] 你刚刚在信里说
[连贯] 以前你抢了我一百次
[连贯] 那
[连贯] 这第一百零一次
[连贯] 就由我来抢吧
[不连贯] 二十
[连贯] 我要从全世界的女人手中
[连贯] 把你抢走
[不连贯] 我要陪伴你一生一世
[连贯] 嫁给我吧
[不连贯] 我答应你
[不连贯] 这回你可以嫁给我了吧
[不连贯] 给我戴上
[不连贯] 好了
[连贯] 恭喜你们
[不连贯] 这就是双向奔赴的爱情吧
[不连贯] 这就是双向奔赴的爱情吧
[不连贯] 这狗粮
[连贯] 我估计能吃一年
[不连贯] 什么一年哪
[连贯] 是一生一世好吗
[不连贯] 回家
[连贯] 你今天真帅
[不连贯] 谢谢盛太太
[连贯] 我愿用盛世的繁华
[连贯] 换取一生的阑珊

【例3】
[不连贯] 乾坤之力
[连贯] 请帮崔尘解除惜澜花之咒
[不连贯] 坦坦
[不连贯] 坦坦
[不连贯] 坦坦
[不连贯] 崔先生 贵公司推出的能源产品
[连贯] 在新能源汽车的运用上非常成功
[不连贯] 请问您未来是
[连贯] 继续打算进军汽车市场
[连贯] 还是发展新的事业版图呢
[不连贯] 汽车行业
[连贯] 会一直是我司的基本盘
[不连贯] 对新领域的发展
[连贯] 我们也一直持开放的态度
[连贯] 打造高效 系统化的技术体系
[连贯] 支持环保事业的发展
[不连贯] 来了来了来了
[连贯] 崔先生来了
[不连贯] 那是他的车吗
[不连贯] 是吧
[不连贯] 是
[不连贯] 崔总
[不连贯] 崔先生
[不连贯] 来了来了
[不连贯] 快快快
[连贯] 快准备好
[不连贯] 跟上 跟上
[不连贯] 拍照 拍照
[不连贯] 拍照 拍照
[不连贯] 崔先生
[不连贯] 不要拍照
[连贯] 往后退
[不连贯] 借过一下 借过一下
[连贯] 借过
[不连贯] 不好意思啊 我们不接受采访

以上是对不同语言影视节目中字幕台词进行切分的例子，请你根据上面的例子完成下面的{}台词切分。要求：
1.请仿照前面的例子，根据语义连贯性与完整性判断提供的每一句台词与前面一句台词是否连贯；
2.提供的待切分台词中可能涉及一个或多个角色的对话，将说话人的切换作为切分的辅助参考，即使同一说话人说了多句台词，也可将这些台词切分为多组；
3.以`[连贯]`和`[不连贯]`来标记切分的结果且保证切分结果的行数与提供的台词行数完全一致，不要做任何台词的合并与拆分；
4.以与例子一致的格式来输出切分结果，仅输出切分结果，不要输出任何其他多余的内容。

以下为待切分的{}台词：
{}

切分结果：
"""

sentence_segment_template_en = """【例1】
[不连贯] We use their division against them.
[不连贯] Pin the attack on Jinx,
[连贯] post a reward too substantial to ignore.
[不连贯] I'm sorry, Mel. I'm not comfortable trusting our fates to chance.
[不连贯] Jinx has proven elusive.
[不连贯] Our healing can only begin once she's been brought to justice.
[不连贯] Then it's settled, 2 to 1.
[连贯] We invade.
[不连贯] If I may? In crises such as these,
[连贯] it's imperative you present a unified front to the public,
[连贯] whatever your personal feelings.
[不连贯] How wise.
[不连贯] I'll agree to endorse the invasion.
[连贯] But I draw the line at Hextech.
[不连贯] We have an ethos.
[连贯] Such force must be a final resort.
[不连贯] Agreed.
[不连贯] Then it's settled, 2 to 1.
[不连贯] I had the shot.
[不连贯] Your mother left this to you.
[不连贯] The Kiramman Key?
[连贯] No, I'm not…
[连贯] I don't deserve it.
[不连贯] It's your legacy now, Caitlyn.
[不连贯] What is she still doing here?
[不连贯] Hey.
[连贯] You were right, Cait.
[连贯] Powder's gone.
[不连贯] I can make this right.
[连贯] If you get Jayce to fix the gauntlets, I can do this myself.

【例2】
[不连贯] Anyway… Okay.
[连贯] Now, if you kids get bored, I found some old records and games of your dad's,
[连贯] and a few magazines that I don't think he wanted me to see,
[连贯] but I got rid of them.
[不连贯] They were all redheads.
[不连贯] He sure has a type!
[不连贯] Anyhoo, you guys have fun.
[不连贯] Whoa, this must be from a Nintendo Zero.
[不连贯] No way.
[不连贯] Your grandma just hooked us up.
[不连贯] -No way. -Hell yeah.
[连贯] Totally.
[不连贯] What is it?
[连贯] Best summer ever!
[不连贯] Sign me up!
[不连贯] But seriously, what is it?
[不连贯] One, two, three, four!
[不连贯] It's the summer. Let's have fun. We should do whatever we want. Right?
[连贯] Wrong.
[不连贯] -You know what I want? -Don't say Jay.
[连贯] Jay.
[不连贯] -You are banned from this house. Get out. -No!
[不连贯] Donna, it's Kitty! Your daughter is about to have sex!
[连贯] Where is she?
[不连贯] Wait, so Dad wasn't as smooth as he says he was?
[不连贯] Let's just say you're lucky you exist.
[不连贯] Hey, which one of you yelled at my kids?
[连贯] I don't know which ones are yours, but they deserved it.
[连贯] That's Gwen and Nate, and thanks.
[不连贯] I was pregnant with him for ten months.
[连贯] The doctor said it was okay, but I don't think it was.
[不连贯] I think you might be having a reaction to one of my Lip Smackers.
[连贯] Really?
[不连贯] -I want to come out to your grandmother. -I have an interview tomorrow.
[不连贯] -Do you have any references? -Oh, my bosses all loved me.

【例3】
[不连贯] She gets moody.
[连贯] And now she needs a nap.
[连贯] Sleep!
[不连贯] Train, say your prayers, eat your vitamins.
[连贯] Chewables and the swallow kind.
[不连贯] I'm coming for you, Hep Cat.
[不连贯] Body slam!
[不连贯] Be true to yourself. True to your country.
[连贯] Be a real American. Hoo.
[不连贯] Okay. Five minutes, everybody.
[不连贯] Hey!
[不连贯] - I thought she was cut? - Yeah.
[连贯] That's what I thought.
[不连贯] You're wrong about me.
[不连贯] Yeah, I doubt it.
[不连贯] Damn, girl.
[不连贯] How long must this... go on?
[连贯] This punishment?
[连贯] Haven't I served my term?
[不连贯] Can't I apply... for a pardon?
[不连贯] You know what I feel like?
[连贯] I feel... all the time...
[连贯] like a cat...
[连贯] ...on a hot tin roof.
[不连贯] Who has come here...
[连贯] to save...
[连贯] you...
[连贯] and you...
[连贯] and you...
[连贯] from evil.
[不连贯] Train. Say your prayers. Eat your vitamins.
[不连贯] Cause, uh...
[不连贯] -Oh... - You fucking bitch!
[连贯] Oh, you fucking cunt! I should fucking kill you!
[不连贯] -What are you doing here? -Don't play dumb, homewrecker!
[连贯] Husband-fucker!

以上是对不同语言影视节目中字幕台词进行切分的例子，请你根据上面的例子完成下面的{}台词切分。要求：
1.请仿照前面的例子，根据语义连贯性与完整性判断提供的每一句台词与前面一句台词是否连贯；
2.提供的待切分台词中可能涉及一个或多个角色的对话，将说话人的切换作为切分的辅助参考，即使同一说话人说了多句台词，也可将这些台词切分为多组；
3.以`[连贯]`和`[不连贯]`来标记切分的结果且保证切分结果的行数与提供的台词行数完全一致，不要做任何台词的合并与拆分；
4.以与例子一致的格式来输出切分结果，仅输出切分结果，不要输出任何其他多余的内容。

以下为待切分的{}台词：
{}

切分结果：
"""

sentence_segment_template_ja = """【例1】
[不连贯] 火もだいぶ治まってきた
[不连贯] マーレ兵は ほぼ全滅して
[连贯] 無垢の巨人は もういない
[连贯] ひとまず ここは安全だ
[不连贯] そうか
[连贯] そろそろ俺たちに 例の白昼夢の話をしてくれ
[不连贯] エレンは 今 何をしている？
[不连贯] 見てのとおりだ
[不连贯] そうか
[连贯] 俺の故郷もダメなのか？
[不连贯] フロック！　大丈夫か？
[不连贯] 俺が壁の崩壊で死にかけてる間
[连贯] 巨人討伐を指揮してたらしいな
[不连贯] ジャン
[连贯] 生きてたか
[不连贯] ああ 心配してくれてありがとう
[不连贯] だが くたばるわけには いかねえだろ
[连贯] エルディア帝国が 復活するって瞬間なのに
[不连贯] イェレナ
[不连贯] 義勇兵を集めろ 全員拘束する
[不连贯] なんだと？
[不连贯] ご無事で何よりです ブラウスさん
[不连贯] 君たちも無事でよかった
[不连贯] しかし この状況で呼び出して すまない
[不连贯] 君は… なぜここに？
[不连贯] 信じて もう争う気はないの
[不连贯] 私は ただ… ファルコを返してほしいだけ
[连贯] そしたら私たちは どこかに消えるから
[不连贯] ファルコは どこ？

【例2】
[不连贯] ポルコ お話聞かせて
[连贯] 今度 二人っきりの時にな
[不连贯] あのアメリカさん おかしいの
[连贯] わたしの顔を見るなり “けっこんしてくれ”だって
[连贯] だから教えてあげたわ
[不连贯] わたしは３回 飛行艇乗りとけっこんしたけど
[连贯] 一人は戦争で 一人は大西洋で
[连贯] 最後の一人はアジアで死んだって…
[不连贯] わかったのか
[不连贯] 今日 れんらくがあったの
[连贯] ベンガルのおく地で ざんがいが見つかったって
[不连贯] ３年待ったわ
[连贯] もうなみだも かれちゃった
[不连贯] いいやつはみんな死ぬ
[连贯] 友へ…
[不连贯] マルコ ありがとう いつもそばにいてくれて
[连贯] もう あなただけになっちゃったわね 古い仲間は…
[不连贯] この店で ひとつだけ気にくわねえのは
[不连贯] あの写真をはずさねえことだ
[连贯] ダメよ　破いちゃ
[连贯] マルコが人間だった時の たった一枚だけ残った写真なんだから
[不连贯] どうやったら あなたにかけられた まほうがとけるのかしらね
[不连贯] あのアメリカやろう いいうでしてるぜ…
[连贯] うらやましい
[连贯] わたしも このくらいかせいでみたいですな
[不连贯] 今月のはらいだ
[连贯] 飛行艇のローンは終わりました
[不连贯] いかがでしょう
[连贯] 愛国債権などを お求めになって 民族に貢献されては

【例3】
[不连贯] どのくらいの人間がいるか 知ってる？
[连贯] 67億人だよ
[不连贯] 67億人？
[连贯] 君たちは？
[连贯] 知らないわ
[连贯] もう 何人かしかいないんだよね
[不连贯] ぼくも母に聞くまでは⸺
[连贯] 君たちのこと知らなかった
[不连贯] これまでにも多くの生き物が 絶滅してきた
[连贯] ぼくも本でしか見たことないけど
[不连贯] 美しい種族たちが⸺
[连贯] 地球の環境の変化に 対応できなくて⸺
[连贯] 滅んでいった
[不连贯] 残酷だけど
[连贯] 君たちも そういう運命なんだ
[不连贯] 運命ですって？
[连贯] あなたが余計なことをしたから
[连贯] 私たちは ここを出ていくことに なったのよ
[不连贯] 何としても生きのびなきゃ いけないってお父さんも言ってた
[连贯] だから危険があっても 新しいところへ行くの
[不连贯] そうやって私たちの種族が⸺
[连贯] どこかで工夫して 暮らしているのを⸺
[连贯] あなたたちが知らないだけよ
[连贯] 私たちは⸺
[连贯] そう簡単に滅びたりしないわ！
[不连贯] ごめん
[连贯] 君の言うとおりだよ
[连贯] 本当は 死ぬのはぼくの方だ
[不连贯] え…
[连贯] ここが良くなくて…
[连贯] 来週　手術するけど
[连贯] きっとだめだ
[不连贯] 心臓？
[不连贯] 小さい時から病気で
[连贯] 何もできなかったから
[连贯] 君を見た時
[连贯] 守ってあげられたらと 思ったんだけど
[连贯] やっぱりだめだった
[连贯] 本当に ごめん

以上是对不同语言影视节目中字幕台词进行切分的例子，请你根据上面的例子完成下面的{}台词切分。要求：
1.请仿照前面的例子，根据语义连贯性与完整性判断提供的每一句台词与前面一句台词是否连贯；
2.提供的待切分台词中可能涉及一个或多个角色的对话，将说话人的切换作为切分的辅助参考，即使同一说话人说了多句台词，也可将这些台词切分为多组；
3.以`[连贯]`和`[不连贯]`来标记切分的结果且保证切分结果的行数与提供的台词行数完全一致，不要做任何台词的合并与拆分；
4.以与例子一致的格式来输出切分结果，仅输出切分结果，不要输出任何其他多余的内容。

以下为待切分的{}台词：
{}

切分结果：
"""

sentence_segment_template_ko = """【例1】
[不连贯] 아, 우리 아빠 이렇게 아픈데 나만 좋자고
[不连贯] 미안해, 아빠
[不连贯] 안녕하세요
[不连贯] 아, 저, 정수범 님 이제 들어가겠습니다
[不连贯] 별일 없으셨죠?
[不连贯] 기분 좀 어떠세요?
[不连贯] 수술 설명 좀 해 드릴게요
[连贯] 말씀드렸지만 균종 제거하고 남아 있는 조직으로 성형을 하면
[连贯] 시간은 한 5시간 정도 걸릴 것 같은데
[连贯] 상황이 안 좋아지면 더 걸릴 수도 있고요
[不连贯] 선생님, 그래도 자주 하시는 수술이죠?
[不连贯] 네
[不连贯] 그럼 크게 걱정 안 해도 되는 거죠?
[不连贯] 생명에 지장이 있다거나 그런 수술은 아닌 거죠?
[不连贯] 그건 장담 못 드리는 거고요
[连贯] 수술할 땐 워낙 변수들이 많아서요
[连贯] 아무튼 최선을 다하겠습니다
[不连贯] 으이구
[连贯] 그냥 무조건 괜찮다 그렇게 말씀해 주시지
[不连贯] 정수범 환자 따님 오늘 결혼해요
[连贯] 가뜩이나 가족들이 서로 미안하고 그럴 텐데
[不连贯] 말씀이라도 이쁘게 잘해 주시지
[连贯] 뭐, 말이라도
[连贯] '무조건 수술 성공할 거니까 아무 걱정 하지 마라'
[不连贯] 이렇게 말씀해 주시면 안 돼요?
[不连贯] 교수님한텐 별 어려운 수술도 아니잖아요
[不连贯] 그러다 정수범 환자 페리애뉼라 업세스면?
[连贯] 수술하다 엠볼리즘 돼서 브레인 데미지 받으면 어떡할 건데?
[不连贯] 에이, 그, 그럴 확률 낮잖아요
[不连贯] 아휴, 사람이 감정이 없어
[不连贯] 감정이 있으면 그게 수술하는 데 도움이 돼?
[连贯] 도움 되면 하고, 안 돼, 근데
[连贯] 하나도 도움 안 돼
[不连贯] 죄송합니다
[不连贯] 너 앞으로 절대 환자들한테 하지 마

【例2】
[不连贯] 네, 진짜 지아가 받게 됐어요
[不连贯] 도너는 22세 여자로 49kg고요
[连贯] LFT는 40 미만으로 괜찮습니다
[连贯] 강운대병원에 7일간 입원하셨다고 합니다
[不连贯] 그럼 저희도 바로 수술 준비하겠습니다
[连贯] 네
[不连贯] 어, 어떡해
[连贯] 감사합니다, 선생님
[连贯] 감사합니다, 선생님, 정말
[连贯] 아, 감사합니다, 선생님
[连贯] 정말 감사합니다, 선생님
[不连贯] 아직 뇌사 판정 위원회의 절차도 남았고
[连贯] 도너, 그러니까 간을 주시는 분의 간 상태도 봐야 하지만
[连贯] 큰 문제는 없을 거 같습니다
[连贯] 빠르면 오늘 밤에 바로 수술할 수 있어요
[不连贯] 그럼, 우리 지아 그럼
[连贯] 이제 살 수 있는 거죠, 선생님?
[连贯] 네
[不连贯] 감사합니다, 감사합니다, 정말
[不连贯] 간 상태 어때요? 출발했어요?
[不连贯] 교수님, 어떡하죠?
[连贯] 무슨 일 있어요?
[连贯] 도너 간이 너무 두껍습니다
[连贯] 간이 500g은 넘어 보이고 AP도 8cm 정도 됩니다
[连贯] 어떡하죠, 교수님?
[连贯] 너무 커서 안 될 거 같습니다, 교수님
[不连贯] 마지막 방법이자 유일한 방법이니까
[连贯] 아기를 살리기 위한
[不连贯] 출혈 많습니다, 선생님
[不连贯] 다음 EVD 생기면 네가 한번 해 볼래?
[连贯] 네?
[不连贯] 저 집도하기로 했습니다
[连贯] 저 할 때보다 더 떨려요
[不连贯] 포기하시는 게...
[不连贯] 잘못된 건 아니겠지?

【例3】
[不连贯] 정말 많은 취재진이 지금 보이십니까?
[连贯] 말씀드리는 순간 배우 천송이 씨가 등장했습니다
[不连贯] 가시죠, 가시죠
[不连贯] 왜 사랑을 믿지 않느냐고요?
[连贯] 완전 믿죠
[不连贯] 옛날에도 그런 사람들 많았어요
[连贯] 신분을 뛰어넘는 사랑을 한다면서
[连贯] 마님이랑 머슴이 바람나서 밤도망 가고
[连贯] 그런데 목숨 걸었던 그 사랑은 3년도 못 가서 끝이 나요
[连贯] 마님은 호미 들고 바람난 머슴 잡으러 돌아다니고
[不连贯] 프러포즈요?
[连贯] 나만을 위해서 노래를 불러 주는 거?
[连贯] 무릎 꿇고
[连贯] 무릎 꿇고 노래 부르며 결혼을 구걸하는 인간들
[不连贯] 유치하죠
[连贯] 유치한 맛으로 하는 거죠 사랑이란 게
[不连贯] 사랑은 시간을 못 이깁니다
[不连贯] 모든 걸 다 이기는 사랑
[连贯] 저도 곧 만날 거 같아요
[连贯] 만나면 저도 가겠죠
[连贯] 시집갈 때 됐잖아요
[不连贯] 결국 그 구두를 신고 그 배를 탔다 이 말입니까?
[不连贯] 그분에게 정말 나쁜 일이 생기면
[连贯] 아무렇지 않을 자신 있으십니까?
[不连贯] 송이가 없어졌어요 아무리 찾아도 없어
[不连贯] 아, 그걸 어떻게 믿냐고!
[不连贯] 이 남자가 사라질 때 천송이 씨도 사라졌어요
[不连贯] 나 그날 정말 이상한 일이 있었어
[连贯] 혹시 그날 거기 왔었나?

以上是对不同语言影视节目中字幕台词进行切分的例子，请你根据上面的例子完成下面的{}台词切分。要求：
1.请仿照前面的例子，根据语义连贯性与完整性判断提供的每一句台词与前面一句台词是否连贯；
2.提供的待切分台词中可能涉及一个或多个角色的对话，将说话人的切换作为切分的辅助参考，即使同一说话人说了多句台词，也可将这些台词切分为多组；
3.以`[连贯]`和`[不连贯]`来标记切分的结果且保证切分结果的行数与提供的台词行数完全一致，不要做任何台词的合并与拆分；
4.以与例子一致的格式来输出切分结果，仅输出切分结果，不要输出任何其他多余的内容。

以下为待切分的{}台词：
{}

切分结果："""

vividness_fewshot = {
    'score format':'{"A": score, "B": score, "C": score, ...}',
    'en2zh': """<样例>
英语原文：
[上下文] As one of Piltover's founders,
[上下文] what haven't you to show for your remarkable life?
[上下文] You should be proud of what you've accomplished, Viktor.
[上下文] Figments.
[上下文] My contributions will be short-lived, even in your memory.
[待评估] I have seen many students.
[待评估] It's a sad truth that
[待评估] those who shine brightest often burn fastest.
[上下文] I didn't know you were an artist.
[上下文] Hm. There's quite a lot about me you don't know.
[上下文] Listen, I'm sorry for disappearing last night.
[上下文] Duty calls.
[上下文] Viktor's dying.

中文翻译：
译文A：
我见过许多学生
可悲的是
最耀眼的星辰往往最先燃尽

译文B：
我目睹过众多学子
有个悲哀的真相
光芒最盛者往往最先陨落

译文C：
我见过的学生很多
令人心痛的是 越是光芒万丈的人
越会更快地燃尽生命的能量

译文D：
我教过这么多学生
说句实在话
越聪明的孩子越容易栽跟头

译文E：
我见过很多学生
一个悲哀的事实是
那些最闪耀的人往往燃烧得最快

评估得分：        
{"A": 88, "B": 85, "C": 92, "D": 55, "E": 78}""",
    'zh2en': """<样例>
中文原文：
[上下文] 被你到处践踏的遗迹
[上下文] 可是无价之宝
[上下文] 是无法用价值来衡量的珍宝
[待评估] 历史虽然会重演
[待评估] 但是人类是无法回到过去的
[上下文] 看来你是不会明白的
[上下文] 明...明白...我明白了
[上下文] 再也...不会...那么做了...请原谅
[上下文] 不能原谅

英语翻译：
译文A：
History repeats,
but we can’t go back to what was.

译文B：
History might replay,
but mankind cannot go back in time.

译文C：
Even if history repeats,
the past remains forever inaccessible to us.

译文D：
Although history may repeat itself,
humans cannot return to the past.

译文E：
History often echoes,
yet there’s no way for us to turn back the clock.

译文F：
Although the history may be replayed over and over again,
human beings can never go back to the past.

译文G：
Although history will always repeat itself,
humans can't return to their past.

评估得分：
{"A": 70, "B": 85, "C": 75, "D": 82, "E": 92, "F": 55, "G": 78}""",
    'ja2zh': """<样例>
日语原文：
[上下文] でもさ、里美が三上と付き合ってるって知った時...
[上下文] 俺、どうすればいいか分かんなかった
[上下文] 完治くんはね、恋愛ってスポーツみたいに考えてる
[上下文] 勝ち負けがあるって思ってる
[上下文] え？
[待评估] 本当の愛って、相手を好きになるだけじゃないの
[待评估] 相手を好きになった自分のことも好きになることなのよ！
[上下文] 意味が分かんないよ、それ
[上下文] つまりね——
[上下文] 愛されてる実感がないから
[上下文] 完治くんはいつもビクビクしてるの！

中文翻译：
译文A：
真正的爱不只是喜欢对方
更是要喜欢上那个喜欢着对方的自己呀！

译文B：
爱不仅是倾心于谁
更是爱上因他而熠熠生辉的自己

译文C：
真爱啊 不光是你有多喜欢那个人
更是你会喜欢那个喜欢着他的自己！

译文D：
真正的爱 不只是喜欢上对方
而是连那个喜欢上对方的自己也一并喜欢上了啊！

译文E：
当你爱上一个人时
连带着也会爱上此刻陷入爱情的自己——这才是爱

评估得分：        
{"A": 87, "B": 83, "C": 91, "D": 93, "E": 76}""",
    'en2de': """<样例>
英语原文：
[上下文] As one of Piltover's founders,
[上下文] what haven't you to show for your remarkable life?
[上下文] You should be proud of what you've accomplished, Viktor.
[上下文] Figments.
[上下文] My contributions will be short-lived, even in your memory.
[待评估] I have seen many students.
[待评估] It's a sad truth that
[待评估] those who shine brightest often burn fastest.
[上下文] I didn't know you were an artist.
[上下文] Hm. There's quite a lot about me you don't know.
[上下文] Listen, I'm sorry for disappearing last night.
[上下文] Duty calls.
[上下文] Viktor's dying.

德语翻译：
译文A：
Ich habe viele Schüler gesehen.
Es ist eine traurige Wahrheit:
Die hellsten Sterne verglühen am schnellsten.

译文B：
So viele Schüler habe ich begleitet.
Doch das bittere Gesetz lautet:
Je strahlender das Talent, desto kürzer die Zeit.

译文C：
In meiner Laufbahn sah ich unzählige Schüler.
Das Tragische daran:
Gerade die Begabtesten brennen sich am schnellsten aus.

译文D：
Ich hatte viele Studenten.
Ehrlich gesagt:
Die Klügsten stolpern am leichtesten.

译文E：
Viele Schüler sind mir begegnet.
Doch das Schicksal will es:
Die hellsten Flammen erlöschen zuerst.

评估得分：        
{"A": 85, "B": 88, "C": 90, "D": 60, "E": 82}""",
    'en2fr': """<样例>
英语原文：
[上下文] As one of Piltover's founders,
[上下文] what haven't you to show for your remarkable life?
[上下文] You should be proud of what you've accomplished, Viktor.
[上下文] Figments.
[上下文] My contributions will be short-lived, even in your memory.
[待评估] I have seen many students.
[待评估] It's a sad truth that
[待评估] those who shine brightest often burn fastest.
[上下文] I didn't know you were an artist.
[上下文] Hm. There's quite a lot about me you don't know.
[上下文] Listen, I'm sorry for disappearing last night.
[上下文] Duty calls.
[上下文] Viktor's dying.

法语翻译：
译文A：
J’ai vu tant d’étudiants.  
C’est une triste réalité :  
ceux qui brillent le plus s’éteignent souvent le plus vite.  

译文B：
J’ai connu beaucoup d’élèves.  
Hélas, c’est une loi cruelle :  
les esprits les plus brillants se consument aussi les premiers.  

译文C：
J’ai enseigné à maints jeunes talents.  
Mais le destin est ironique :  
plus la flamme est vive, plus elle se consume rapidement.  

译文D：
Tant d’étudiants ont croisé ma route.  
Une vérité douloureuse demeure :  
les étoiles les plus brillantes sont les premières à disparaître.  

译文E：
J’ai formé nombre d’élèves.  
Et toujours, même constat amer :  
les plus prometteurs partent trop tôt.  

评估得分：        
{"A": 86, "B": 92, "C": 88, "D": 90, "E": 80}""",
    'zh2th': """<样例>
中文原文：
[上下文] 被你到处践踏的遗迹
[上下文] 可是无价之宝
[上下文] 是无法用价值来衡量的珍宝
[待评估] 历史虽然会重演
[待评估] 但是人类是无法回到过去的
[上下文] 看来你是不会明白的
[上下文] 明...明白...我明白了
[上下文] 再也...不会...那么做了...请原谅
[上下文] 不能原谅

泰语翻译：
译文A：
ประวัติศาสตร์อาจซ้ำรอยเดิม  
แต่มนุษย์ย้อนกลับไปหาอดีตไม่ได้  

译文B：
แม้ว่าประวัติศาสตร์จะวนซ้ำ  
แต่อดีตนั้นมนุษย์ไม่อาจหวนคืน  

译文C：
ถึงประวัติศาสตร์จะเล่าขานซ้ำแล้วซ้ำเล่า  
แต่มนุษยชาติไม่อาจก้าวย้อนกลับไป  

译文D：
แม้กาลเวลาจะหมุนเวียนเหมือนเดิม  
แต่เราก็ไม่อาจกลับสู่ช่วงเวลาที่ผ่านมา  

译文E：
ประวัติศาสตร์อาจจะเดินทางเป็นวงกลม  
แต่เส้นทางของมนุษย์เดินไปข้างหน้าอย่างเดียว  

评估得分：        
{"A": 85, "B": 92, "C": 88, "D": 83, "E": 78}""",
    'zh2vi': """<样例>
中文原文：
[上下文] 被你到处践踏的遗迹
[上下文] 可是无价之宝
[上下文] 是无法用价值来衡量的珍宝
[待评估] 历史虽然会重演
[待评估] 但是人类是无法回到过去的
[上下文] 看来你是不会明白的
[上下文] 明...明白...我明白了
[上下文] 再也...不会...那么做了...请原谅
[上下文] 不能原谅

越南语翻译：
译文A：
Lịch sử dù lặp lại  
Con người vẫn không thể quay về quá khứ

译文B：
Dẫu lịch sử có tuần hoàn  
Nhân loại chẳng bao giờ trở lại được ngày xưa

译文C：
Cho dù lịch sử tái diễn  
Loài người mãi mãi không về nơi cũ

译文D：
Lịch sử có thể đi vòng  
Nhưng nhân gian chỉ tiến về trước

译文E：
Dù thời gian xoay vần  
Vẫn không cách nào níu kéo dĩ vãng

评估得分：        
{"A": 84, "B": 93, "C": 89, "D": 76, "E": 87}""",
    'zh2es': """<样例>
中文原文：
[上下文] 被你到处践踏的遗迹
[上下文] 可是无价之宝
[上下文] 是无法用价值来衡量的珍宝
[待评估] 历史虽然会重演
[待评估] 但是人类是无法回到过去的
[上下文] 看来你是不会明白的
[上下文] 明...明白...我明白了
[上下文] 再也...不会...那么做了...请原谅
[上下文] 不能原谅

西班牙语翻译：
译文A：
Aunque la historia se repita,
la humanidad no puede volver al pasado.

译文B：
Si bien la historia es cíclica,
el ser humano jamás regresará a lo que fue.

译文C：
Por mucho que se repita la historia,
los hombres no tenemos vuelta atrás.

译文D：
La historia puede repetirse,
pero el reloj del tiempo no retrocede.

译文E：
Aunque la historia vuelva a ocurrir,
nunca podremos recuperar lo perdido.

评估得分：        
{"A": 85, "B": 94, "C": 89, "D": 82, "E": 87}""",
    'ko2zh': """<样例>
韩语原文：
[上下文] 넌 왜 자꾸 나를 피해?
[上下文] 내가 무서워?
[上下文] 아니... 무서운 건 너야
[待评估] 네가 웃을 때마다
[待评估] 내 마음이 조각조각 흩어져
[待评估] 다시 주워 담을 수가 없어
[上下文] 미안해...
[上下文] 이렇게 밖에 할 수 없어서
[上下文] 나도... 너무 아파

中文翻译：
译文A：
每次你笑的时候
我的心就碎成一片片
再也拼不回来了

译文B：
你每绽放一次笑容
我的心就四散零落
再也无法完整拾起

译文C：
当你微笑时
我的心就支离破碎
再也无法复原

译文D：
你的每个笑容
都让我的心碎落满地
再也无法收拾

译文E：
看你笑的时候
我的心就一片片剥落
再也拼凑不回

评估得分：        
{"A": 88, "B": 92, "C": 85, "D": 90, "E": 87}"""
}

vividness_preference_template = """【{}到{}台词翻译生动性评估】

<要求>
请对以下{}到{}台词翻译的多个不同译文进行生动性打分，打分采用0到100的整数评分。待评估{}原文会提供前面或后面的多句上下文以供参考，待评估台词和上下文台词分别由'[待评估]'和'[上下文]'标记。台词翻译生动性评分原则如下：
原则1（译文准确性评估）：译文可以有一定的意译，不过也需保证一定的翻译准确性，能够准确传达原文含义且不能多译漏译（权重：30%）；
原则2（译文口语化评估）：评估译文是否口语化，是否符合上下文的对话场景，是否能够传达原文的情绪、氛围、语气等信息（权重：30%）；
原则3（译文表现力评估）：评估译文是否具有表现力，是否措辞生动，情感是否丰富，是否更容易打动观众，可以有一定的文学性（权重：40%）。

评分标准（生动性）：
100：翻译极具表现力和感染力，情感丰富，能够强烈打动观众。
50：翻译具有一定表现力，情感表达较为平淡，但仍能传递部分情感。
0：译文翻译失准且缺乏表现力和情感，无法引起观众的任何情感共鸣。

{}

<任务>
{}原文：
{}

{}翻译：
{}

注意，你需要以json格式输出评分：{}
评估得分（只需输出评分即可，不要输出其他内容）："""