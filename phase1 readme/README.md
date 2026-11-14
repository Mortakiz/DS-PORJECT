## 1.Team and roles: 
1- Yaman Al-Hindawi: Engineering lead

2- Soode Al-Rosan: Evaluation lead, Data lead

3- Mohammad Al-Omari: Product lead

## 2. Users, and the problem we will fix:

Users:  First and second year students at Jordan University of Science and Technology.

The problem: In each term first and second year students struggle with what courses should they use, what are the prerequisites for this course? how many courses this course is a prerequisite for? SO! we decided to make an application that can help our students with their academic plan

how? basically instead of searching through the university's webstite and look through scattered information and ask in facebook groups they can just ask a LLM and it will answer them right away!

## 3. What the system will do?

It's a qeustion and answer assistant! it will retrieve JUST'S offical study plan for any major like Data Science, Computer Science... etc

The user will ask it a question like: 

"ما هو المتطلب السابق لمساق  التعلم العميق؟"

The system will immediately answer: 

"المتطلب السابق لمساق التعلم العميق هو: تعلم الالة"

another example: 

"كم عدد ساعات مساق تصور و اكتشاف البيانات؟"

answer:

"عدد ساعات مساق تصور و اكتشاف البيانات هو 3 ساعات"

## 4. Data:

We will use the official JUST documents like:

study plans and Regulations (University registration rules and guidelines from the JUST regulations page.)

Amount: 1 complete data science study plan (29 pages, ~150 course records)

## 5. Planned architecture:

user asks a question -> Search our study plan data (find relevant courses) -> Send the question + the found course rules to an LLM API ->
Return the answer

## 6. How you will measure success:

Answer accuracy: The percentage of correct answers (30-question seed test set).

Citation accuracy: The percentage of answers that correctly cite the exact course code or page number.

Refusal correctness: The percentage of "should refuse" questions where the system correctly says "I don't know" instead of hallucinating.

## 7. Risks:

Compute limits, The free API we will use may hit the rate limit

backup plan: 

We will cache API responses during development (save answers locally) so we don't waste API calls, and use Google Colab as a backup environment.

## Data Card

short description of the data that we will use

## Source:

Study Plans: Jordan University of Science and Technology (JUST) — Faculty of Computer and
Information Technology. 

Link:https://www.just.edu.jo/ar/FacultiesandDepartments/it/Pages/Programs.aspx?degree =1

## Permission:

These are publicly available documents on the official JUST website. No login is required. They
are published for public use by students

## Size:

1 complete study plan (data science) (29 pages) 14,500 tokens.

## Language and format:

Arabic, The original format is HTML tables saved as a PDF For the project, then we will extract
this text from the PDF and turn it into a clean JSON/CSV format so the system can read it
perfectly

## Sensitive content:

None. The documents contain only public academic rules, course descriptions, and
prerequisites. No personal student data (e.g., names, grades, IDs) is included in the source files.

## Sample:

You can find it in the repo's files "data/sample/samples.json"