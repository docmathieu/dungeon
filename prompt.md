

Ci-dessous la description complète de mon objectif :

Objectif général :
-	Créer un jeu simple sans coder moi-même, déléguer tout le code et les tests à l’IA
-	Utiliser le système de skills pour la partie Agent
-	Utiliser python dans sa version la plus récente (équivalent LTS de java), et le composant graphique le plus performant de python pour gérer un grand nombre de pixels, et le plus pertinent pour une future utilisation via des threads.

L’interface de jeu :
-	Couleur de fond noire
-	Le terrain correspondant à un tableau de 10 cases par 10 case, chaque case faisant 10 pixels par 10 pixels
-	Ajouter un trait de 1px entre chaque case du tableau pour plus de lisibilité
-	Au-dessus du terrain :
o	Un champs texte « déplacements » avec la valeur initiale de 0
o	Un champs texte « note » avec la valeur initiale de 0
o	Un champs texte « Information » vide au démarrage
-	En dessous du terrain :
o	Un bouton en dessous du tableau « restart »
o	Un champ inputText « instruct »
o	Un bouton en dessous du tableau « start »

Type de cases du terrain :
-	Case verte: représente l’herbe.
o	C’est le type par défaut des cases
-	Case grise: représente la Pierre.
o	30% des cases de ce type remplacent l’herbe
o	Ces cases sont infranchissables, un déplacement vers ce type de case n’a pas d’effet
-	Case bleue: représente l’eau.
o	20% des cases de ce type remplacent l'herbe
o	Ces cases sont franchissables mais comptent comme deux déplacements

Eléments de jeu :
-	Le personnage :
o	Dessin très simple d’un personnage jaune placé aléatoirement sur une case herbe.
-	La sortie :
o	Dessin très simple d’une porte jaune placée aléatoirement sur une case herbe.

Actions utilisateur :
-	Entrer une suite de déplacements avec les flèches du clavier dans le champs « instruct » :
o	Caractère ← pour un déplacement à gauche
o	Caractère ↑ pour un déplacement en haut
o	Caractère → pour un déplacement à droite
o	Caractère ↓ pour un déplacement en bas
-	Cliquer sur le bouton « start » ou taper entrée depuis le champs « instruct » pour :
o	Démarrer la simulation
-	Cliquer sur le bouton « restart » pour redémarrer le jeu et donc :
o	Réinitialiser l’interface, générer un nouveau terrain
o	Effacer tous les champs d’affichage et inputText

Simulation :
-	Quand la simulation démarre :
o	Le personnage se déplace selon la séquence du champs « instruct ».
o	Insérer une pause de 0.5 secondes entre chaque déplacement.
o	A chaque déplacement mettre à jour la valeur du champs texte « déplacements »
-	Si le personnage se retrouve sur la même case que l’élément « sortie » alors :
o	Le jeu est terminé.
o	Le champs « note » prend la valeur 1
o	La séquence de mouvement s’arrête
o	Le champs texte « Information » prend la valeur « GAGNE »

Maintenant que tu connais le contexte j’aimerai que tu fasses les choses suivantes :
-	Créer les fichiers AGENTS.md et SKILLS.md nécessaires pour :
o	Générer le code du jeu
o	Générer les tests unitaires du jeu
o	Démarrer les tests unitaires
o	Démarrer le jeu
o	Produire un exécutable Windows autonome
-	Et surtout, que tu me donnes toutes les étapes à faire pour arriver cela en tenant compte du fait que :
o	Je souhaite pouvoir reprendre le tout dans VSC
o	Je souhaite déclencher facilement tout ou partie des agents depuis VSC



