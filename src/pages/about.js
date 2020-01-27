import React from 'react'
import { graphql } from 'gatsby'
import { Resume } from '../components/resume'
import { Layout } from '../layout'
// import { graphql } from 'gatsby'

// import { rhythm } from '../utils/typography'
// import * as Lang from '../constants'

export default ({ data, location }) => {
  const metaData = data.site.siteMetadata
  const { title } = metaData
  // const resumes = data.allMarkdownRemark.edges

  // const resume = resumes
  //   .filter(({ node }) => node.frontmatter.lang === Lang.ENGLISH)
  //   .map(({ node }) => node)[0]

  return (
    <Layout location={location} title={title}>
      <Resume />
    </Layout>
    // <div
    // style={{
    //   marginLeft: `auto`,
    //   marginRight: `auto`,
    //   maxWidth: rhythm(24),
    //   padding: `${rhythm(0.5)} ${rhythm(3 / 4)} ${rhythm(1.5)} ${rhythm(
    //     3 / 4
    //   )}`,
    // }}
    // >
    //   hi!
    // </div>
  )
}

// export const pageQuery = graphql`
//   query {
//     allMarkdownRemark(filter: { frontmatter: { category: { eq: null } } }) {
//       edges {
//         node {
//           id
//           excerpt(pruneLength: 160)
//           html
//           frontmatter {
//             title
//             date(formatString: "MMMM DD, YYYY")
//             lang
//           }
//         }
//       }
//     }
//   }
// `
export const pageQuery = graphql`
  query {
    site {
      siteMetadata {
        title
        author
        siteUrl
      }
    }
  }
`
