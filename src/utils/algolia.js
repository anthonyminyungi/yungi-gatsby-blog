const postQuery = `{
    posts: allMarkdownRemark(filter: {fileAbsolutePath: {regex: "/blog/"}, frontmatter: {category: {ne: null}}}) {
        edges {
          node {
            objectId: id
            excerpt(pruneLength: 100)
            fields {
              slug
            }
            frontmatter {
              date(formatString: "YYYY/MM/DD")
              title
              category
              draft
              showToc
            }
          }
        }
      }
    }

`

const flatten = arr =>
  arr.map(({ node: { frontmatter, ...rest } }) => ({
    ...frontmatter,
    ...rest,
  }))
const settings = { attributesToSnippet: [`excerpt:50`] }

const queries = [
  {
    query: postQuery,
    transformer: ({ data }) => flatten(data.posts.edges),
    indexName: `Posts`,
    settings,
  },
]

module.exports = queries
