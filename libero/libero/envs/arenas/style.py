FLOOR_STYLE = {
    "dark": "dark_floor_texture.png",
    "rustic": "rustic_floor.png",
    "light-gray": "light-gray-floor-tile.png",
    "white-marble": "white_marble_floor.png",
    "wood-plank": "seamless_wood_planks_floor.png",
    "brown-ceramic": "brown_ceramic_tile.png",
    "gray-ceramic": "gray_ceramic_tile.png",
    "tile_grigia_caldera": "tile_grigia_caldera_porcelain_floor.png",
}

WALL_STYLE = {
    "light-blue": "light_blue_wall.png",
    "dark-blue": "dark_blue_wall.png",
    "dark-gray-plaster": "dark_gray_plaster.png",
    "gray-plaster": "gray_plaster.png",
    "dark-green": "dark_green_plaster_wall.png",
    "light-gray-plaster": "light-gray-plaster.png",
    "ceramic": "ceramic.png",
    "white": "white_wall.png",
    "yellow-linen": "yellow_linen_wall_texture.png",
    "red-plastic": "table_red_plastic.png",
    "blue-plastic": "table_blue_plastic.png",
    "green-plastic": "table_green_plastic.png"

}

TABLE_STYLE = {
    "novak-wood": "martin_novak_wood_table.png",
    "light-wood": "table_light_wood2.png",
    "dark-wood": "table_dark_wood.png",
    "metalic": "metalic.png",
    "red-plastic": "table_red_plastic.png",
    "blue-plastic": "table_blue_plastic.png",
    "green-plastic": "table_green_plastic.png"
}

STYLE_MAPPING = {"floor": FLOOR_STYLE, "wall": WALL_STYLE, "table": TABLE_STYLE}


def get_texture_filename(type, style):
    assert type in STYLE_MAPPING.keys()
    assert style in STYLE_MAPPING[type].keys()
    return STYLE_MAPPING[type][style]
