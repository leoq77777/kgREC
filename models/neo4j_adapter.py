from neo4j import GraphDatabase, exceptions
import logging
from typing import Dict, List, Optional

class Neo4jAdapter:
    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger("Neo4jAdapter")
        logging.basicConfig(level=logging.INFO)

    def connect(self) -> bool:
        """连接到Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 验证连接
            with self.driver.session(database=self.database):
                self.logger.info(f"成功连接到Neo4j数据库: {self.uri}")
            return True
        except exceptions.Neo4jError as e:
            self.logger.error(f"Neo4j连接失败: {str(e)}")
            return False

    def close(self) -> None:
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j连接已关闭")

    def clear_database(self) -> bool:
        """清空数据库中的所有节点和关系"""
        if not self.driver:
            self.logger.error("未建立数据库连接，请先调用connect()")
            return False

        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("数据库已清空")
            return True
        except exceptions.Neo4jError as e:
            self.logger.error(f"清空数据库失败: {str(e)}")
            return False

    def create_entity(self,
                     entity_id: str,
                     entity_type: str,
                     properties: Optional[Dict] = None) -> bool:
        """创建实体节点"""
        if not self.driver:
            self.logger.error("未建立数据库连接，请先调用connect()")
            return False

        properties = properties or {}
        # 添加实体ID和类型属性
        properties["id"] = entity_id
        properties["entity_type"] = entity_type

        # 创建Cypher查询
        labels = entity_type.replace(" ", "_").upper()
        prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"CREATE (n:{labels} {{ {prop_str} }})"

        try:
            with self.driver.session(database=self.database) as session:
                session.run(query, **properties)
            self.logger.debug(f"已创建实体: {entity_type}({entity_id})")
            return True
        except exceptions.Neo4jError as e:
            self.logger.error(f"创建实体失败: {str(e)}")
            return False

    def create_relationship(self,
                           source_id: str,
                           target_id: str,
                           relationship_type: str,
                           properties: Optional[Dict] = None) -> bool:
        """创建实体间关系"""
        if not self.driver:
            self.logger.error("未建立数据库连接，请先调用connect()")
            return False

        properties = properties or {}

        # 创建Cypher查询
        rel_type = relationship_type.replace(" ", "_").upper()
        prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()]) if properties else ""
        if prop_str:
            query = f"MATCH (s {{id: $source_id}}), (t {{id: $target_id}}) CREATE (s)-[r:{rel_type} {{ {prop_str} }}]->(t)"
        else:
            query = f"MATCH (s {{id: $source_id}}), (t {{id: $target_id}}) CREATE (s)-[r:{rel_type}]->(t)"

        params = {"source_id": source_id, "target_id": target_id, **properties}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                if result.consume().counters.relationships_created == 0:
                    self.logger.warning(f"未找到源实体或目标实体，无法创建关系: {source_id} -> {target_id}")
                    return False
            self.logger.debug(f"已创建关系: {source_id} -[{relationship_type}]-> {target_id}")
            return True
        except exceptions.Neo4jError as e:
            self.logger.error(f"创建关系失败: {str(e)}")
            return False

    def batch_import_from_networkx(self,
                                 graph: "nx.MultiDiGraph",  # noqa: F821
                                 batch_size: int = 1000) -> Dict:
        """从NetworkX图批量导入数据到Neo4j"""
        if not self.driver:
            self.logger.error("未建立数据库连接，请先调用connect()")
            return {"success": False, "nodes_created": 0, "relationships_created": 0}

        stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "node_errors": 0,
            "relationship_errors": 0
        }

        # 首先导入所有节点
        self.logger.info(f"开始导入{len(graph.nodes)}个节点...")
        for node_id, attrs in graph.nodes(data=True):
            entity_type = attrs.get("type", "Entity")
            if self.create_entity(str(node_id), entity_type, attrs):
                stats["nodes_created"] += 1
            else:
                stats["node_errors"] += 1

        # 然后导入所有关系
        self.logger.info(f"开始导入{len(graph.edges)}条关系...")
        for u, v, attrs in graph.edges(data=True):
            rel_type = attrs.get("type", "RELATED_TO")
            if self.create_relationship(str(u), str(v), rel_type, attrs):
                stats["relationships_created"] += 1
            else:
                stats["relationship_errors"] += 1

        self.logger.info(f"批量导入完成: {stats['nodes_created']}个节点, {stats['relationships_created']}条关系")
        return stats

    def run_cypher_query(self,
                        query: str,
                        parameters: Optional[Dict] = None) -> List[Dict]:
        """运行自定义Cypher查询并返回结果"""
        if not self.driver:
            self.logger.error("未建立数据库连接，请先调用connect()")
            return []

        parameters = parameters or {}
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **parameters)
                return [dict(record) for record in result]
        except exceptions.Neo4jError as e:
            self.logger.error(f"Cypher查询执行失败: {str(e)}")
            return []

if __name__ == "__main__":
    # 示例用法
    neo4j_adapter = Neo4jAdapter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    if neo4j_adapter.connect():
        # 清空数据库（仅示例用）
        neo4j_adapter.clear_database()
        
        # 创建示例实体
        neo4j_adapter.create_entity("user_1", "User", {"name": "John Doe", "age": 30})
        neo4j_adapter.create_entity("movie_1", "Movie", {"title": "Inception", "year": 2010})
        
        # 创建关系
        neo4j_adapter.create_relationship("user_1", "movie_1", "RATED", {"rating": 4.5})
        
        # 查询示例
        results = neo4j_adapter.run_cypher_query(
            "MATCH (u:USER)-[r:RATED]->(m:MOVIE) RETURN u.name, r.rating, m.title"
        )
        print("查询结果:", results)
        
        neo4j_adapter.close()