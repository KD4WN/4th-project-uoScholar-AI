# Create a comprehensive starter TSV user dictionary for university abbreviations / slang.
# Format: <token>\tNNP

entries = [
"컴공", "컴과", "소공", "인지",
"전전컴", "전컴"
"화공",
"건공", "토공", 
"환공", "환원", 
"도공", "교공", "도행"
"생공",

"국문", "영문", "영문과", "중문", "중문과",
"사학", "사복", 

"미대", "시디", "산디", "체대", "음대",

"과사", "과방", "학식", "중도", "전필", "전선", "일선", "교선", "교필", "공소", "선수", 
"랩실", "랩장", "졸논", "졸작"

"공대", "인문대", "정경대", "경영대", "예체대", "자과대", "도과대", "자융대"
]

# Deduplicate while preserving order
seen = set()
deduped = []
for e in entries:
    if e and e not in seen:
        deduped.append(e)
        seen.add(e)

# Write to TSV
path = "/mnt/data/user_dic.tsv"
with open(path, "w", encoding="utf-8") as f:
    for token in deduped:
        f.write(f"{token}\tNNP\n")

path
