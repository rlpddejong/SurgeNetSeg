import numpy as np

#davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'
#davis_palette_np = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3)

youtube_palette = b'\x00\x00\x00\xec_g\xf9\x91W\xfa\xc8c\x99\xc7\x94b\xb3\xb2f\x99\xcc\xc5\x94\xc5\xabyg\xff\xff\xffes~\x0b\x0b\x0b\x0c\x0c\x0c\r\r\r\x0e\x0e\x0e\x0f\x0f\x0f'
youtube_palette_np = np.frombuffer(youtube_palette, dtype=np.uint8).reshape(-1, 3)


### RAMIE Palette ###

custom_palette_np = np.array([
    [  0,   0,   0],            # Background         (No color)
    [  0,   0, 255],            # Azygos             (Blue)   
    [255,   0,   0],            # Aorta              (Red)
    [160, 100, 160],            # Lung               (Pink)
    [255, 160,   0],            # Esophagus          (Orange)
    [255,   0, 157],            # Pericardium        (Purple)
    [255, 255, 255],            # Airways            (White)
    [255, 255,   0],            # Nerves             (Yellow)
    [100,  80,   0],            # Hook               (Orange)
    [128,   0,   0],            # Forceps            (Red)
    [  0, 128,   0],            # Suction/irrigation (Green)
    [  0, 255, 255],            # Vessel sealer      (Cyan)
    [  0, 255,   0],            # Thoracic duct      (Green)
    [  0, 128, 128],            # Clip applier       (Teal)
    [128, 255,   0],            # Needle driver      (Neon Green)
])

custom_names = [
'Background',
'Azygos',
'Aorta',
'Lung',
'Esophagus',
'Pericardium',
'Airways',
'Nerves',
'Hook',
'Forceps',
'Suction/irrigation',
'Vessel sealer',
'Thoracic duct',
'Clip applier',
'Needle driver',
]

custom_palette = custom_palette_np.astype(np.uint8).tobytes()


### SurgeNetSeg Palette ###

color_palette = {
    # Abdomen IDs
    1: (255, 255, 255),  # Tools/camera - White
    2: (0, 0, 255),      # Vein (major) - Blue
    3: (255, 0, 0),      # Artery (major) - Red
    4: (255, 255, 0),    # Nerve (major) - Yellow
    5: (0, 255, 0),      # Small intestine - Green
    6: (0, 200, 100),    # Colon/rectum - Dark Green
    7: (200, 150, 100),  # Abdominal wall - Beige
    8: (250, 150, 100),  # Diaphragm - light Beige
    9: (255, 200, 100),  # Omentum - Light Orange
    10: (180, 0, 0),     # Aorta - Dark Red
    11: (0, 0, 180),     # Vena cava - Dark Blue
    12: (150, 100, 50),  # Liver - Brown
    13: (0, 255, 255),   # Cystic duct - Cyan
    14: (0, 200, 255),   # Gallbladder - Teal
    15: (0, 100, 255),   # Hepatic vein - Light Blue
    16: (255, 150, 50),  # Hepatic ligament - Orange
    17: (255, 220, 200), # Cystic plate - Light Pink
    18: (200, 100, 200), # Stomach - Light Purple
    19: (144, 238, 144), # Ductus choledochus - Light Green
    20: (247, 255, 0),   # Mesenterium
    21: (255, 206, 27),  # Ductus hepaticus - Red
    22: (200, 0, 200),   # Spleen - Purple
    23: (255, 0, 150),   # Uterus - Pink
    24: (255, 100, 200), # Ovary - Light Pink
    25: (200, 100, 255), # Oviduct - Lavender

    # RARP
    26: (150, 0, 100),   # Prostate - Dark Purple
    27: (255, 200, 255), # Urethra - Light Pink
    28: (150, 100, 75),  # Ligated plexus - Brown
    29: (200, 0, 150),   # Seminal vesicles - Magenta
    30: (100, 100, 100), # Catheter - Gray
    31: (255, 150, 255), # Bladder - Light Purple
    32: (100, 200, 255), # Kidney - Light Blue

    # Thorax IDs
    33: (150, 200, 255), # Lung - Light Blue
    34: (0, 150, 255),   # Airway (bronchus/trachea) - Sky Blue
    35: (255, 100, 100), # Esophagus - Salmon
    36: (200, 200, 255), # Pericardium - Pale Blue
    37: (100, 100, 255), # V azygos - Blue
    38: (0, 255, 150),   # Thoracic duct - Green Cyan
    39: (255, 255, 100), # Nerves - Yellow

    # Non-anatomical structures
    40: (150, 150, 150),  # Ureter - Gray
    41: (50, 50, 50),     # Non anatomical structures - Dark Gray
    42: (0, 0, 0),        # Excluded frames - Black

    # Additional structures
    43: (173, 216, 230), # Mesocolon - Light Blue 2
    44: (255, 140, 0),   # Adrenal Gland
    45: (223, 3, 252), # Pancreas (252, 186, 3)
    46: (0, 80, 100), # Duodenum
}

custom_names = {
    1: "Tools/camera",
    2: "Vein (major)",
    3: "Artery (major)",
    4: "Nerve (major)",
    5: "Small intestine",
    6: "Colon/rectum",
    7: "Abdominal wall",
    8: "Diaphragm",
    9: "Omentum",
    10: "Aorta",
    11: "Vena cava",
    12: "Liver",
    13: "Cystic duct",
    14: "Gallbladder",
    15: "Hepatic vein",
    16: "Hepatic ligament",
    17: "Cystic plate",
    18: "Stomach",
    19: "Ductus choledochus",
    20: "Mesenterium",
    21: "Ductus hepaticus",
    22: "Spleen",
    23: "Uterus",
    24: "Ovary",
    25: "Oviduct",
    26: "Prostate",
    27: "Urethra",
    28: "Ligated plexus",
    29: "Seminal vesicles",
    30: "Catheter",
    31: "Bladder",
    32: "Kidney",
    33: "Lung",
    34: "Airway (bronchus/trachea)",
    35: "Esophagus",
    36: "Pericardium",
    37: "V azygos",
    38: "Thoracic duct",
    39: "Nerves",
    40: "Ureter",
    41: "Non anatomical structures",
    42: "Excluded frames",
    43: "Mesocolon",
    44: "Adrenal Gland",
    45: "Pancreas",
    46: "Duodenum",
}

custom_palette_np = np.array([color_palette.get(i, (0, 0, 0)) for i in range(101)])

custom_palette = custom_palette_np.astype(np.uint8).tobytes()