## TSP GA

genetic algorithm TSP

หลักการหลักๆ ของ GA

- มี population (array ของ route tsp)
- assign fitness ให้แต่ละ element ใน population
- selection (เลือก parent ที่จะไปรุ่นต่อไป)
  - ตัวที่ดีสุดจะได้ไปต่อแน่ๆ 1 ตัว
  - tournament selection
    > สุ่มมาหลายๆตัว เลือกตัวเก่งสุดใน tournament
- offspring (สร้างตัวใหม่ๆ จาก parent)

## **init**

ตั้งค่า GA

- pop_size ขนาดของ pop มีกี่ route ใน 1 generation
- gene_size ขนาดของยีน (จำนวนเมือง)

- nextgen_num_parent จำนวน parent ที่จะถูกส่งต่อไป
- tournament_size ขนาดของ tour = route ที่จะถูกนำมาแข่งกันว่าแบบไหนสั้นสุด

> การันตี 1 mutation จากนั้น จะเป็นความน่าจะเป็น ถ้าสุ่มแล้วไม่ได้ทำ = ออก, ถ้าได้ทำ = สุ่มใหม่

- mutation prob ความน่าจะเป็นที่จะทำ mutation อีกครั้ง
- mutation_lim จำนวนครั้งสูงสุดที่จะทำ mutation

## run_ga

ตามหลักการของ GA

## init_pop

สร้าง population ชุดแรก

## assign fitness

คำนวณ fitness

## selection

- เลือกรุ่นต่อไป
- สร้าง arr `next_gen` ไว้เก็บรุ่นต่อไป
- การันตีตัวดีสุด 1 ตัว
- ที่เหลือใช้ tournament

## create_offspring

- สร้าง offspring

- `parent_ind` = index ของ parent ใน next gen ไว้สำหรับให้ numpy random
- `gene_ind` = index ของ gene ไว้สำหรับให้ numpy random

### cross over

- สุ่ม parent 2 ตัวสุ่ม 2 ตำแหน่ง
- cross over ได้รุ่นลูก เก็บไว้ใน next gen

### mutation

- นำรุ่นลูกมาสลับตำแหน่ง gene
- การรันตี 1 mutation ทีเหลือสุ่มว่าได้ทำอีกไหม

## tournament_selection

- วิธีคัดเลือกจาก tournament
  > อาจจะมีความไม่ตรงตามทฤษฎี
- สุ่มเลือก fitness มา n ตัว
- หาตัวที่ fitness ดีสุด
- ไปหา gene ใน pop
