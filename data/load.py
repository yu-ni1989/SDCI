from data.GID.GID_datasets import GIDDataset
from data.potsdam.potsdam_datasets import potsdamDataset
from data.vaihingen.vihengen_datasets import vaihingenDataset
from data.loveDA.loveDA_datasets import loveDADataset
from data.iSAID.iSAID_datasets import iSAID

def load_dataset(cfg, model):
    if cfg.dataset_name == "GID":
        dataset = GIDDataset(img_name_list_path=cfg.img_name_list_path,
                             data_root=cfg.gid_root,
                             img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "potsdam":
        dataset = potsdamDataset(img_name_list_path=cfg.img_name_list_path,
                                 data_root=cfg.potsdam_root,
                                 img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "vaihingen":
        dataset = vaihingenDataset(img_name_list_path=cfg.img_name_list_path,
                                   data_root=cfg.vaihingen_root,
                                   img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "loveDA":
        dataset = loveDADataset(img_name_list_path=cfg.img_name_list_path,
                                data_root=cfg.loveDA_root,
                                img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "iSAID":
        dataset = iSAID(img_name_list_path=cfg.img_name_list_path,
                        data_root=cfg.potsdam_root,
                        img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    else:
        raise NotImplementedError("Unknown dataset")
    return dataset, bg_text_features, fg_text_features
